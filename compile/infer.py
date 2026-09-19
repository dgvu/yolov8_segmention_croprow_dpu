#!/usr/bin/env python3
# infer.py - YOLOv8-seg crop_row on ZCU104
# Supports 2 xmodel output styles:
#   Mode A: raw split DPU outputs: proto + box/cls/msk for strides 8/16/32
#   Mode B: Ultralytics-style outputs: pred [1,37,8400] + proto [1,32,160,160] or NHWC
# Pipeline: ARM preprocess -> DPU -> ARM decode -> union mask -> morphology -> connected components

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import xir
import vart

# -------------------------
# User config
# -------------------------
IMG_SIZE = 640
NC = 1                 # crop_row
NM = 32                # YOLOv8-seg mask coefficients
REG_MAX = 16           # raw split box channels = 4*REG_MAX = 64
STRIDES = [8, 16, 32]

# Candidate thresholds
CONF_THRESH = 0.30
BOX_NMS_IOU = 0.60
MAX_CANDIDATES = 45

# Mask / row extraction thresholds
MASK_THRESH = 0.5           # chuẩn Ultralytics; >0.45 cho biên gọn hơn
MIN_COMPONENT_AREA = 1800   # dùng làm ngưỡng diện tích tối thiểu / instance

# Các biến dưới đây KHÔNG còn dùng từ khi bỏ union+connectedComponents
# (giữ lại để không phá tham số dòng lệnh cũ nếu có script khác import)
MIN_ELONGATION = 1.20
MAX_COMPONENT_AREA_RATIO = 0.75

CLOSE_KERNEL = (5, 5)
OPEN_KERNEL = (3, 3)

COLOR_BGR = (0, 220, 80)
ALPHA = 0.42

# Output buffer mode.
# False: use int8 buffers and manual dequantization. Recommended for ZCU104 DPU.
# True : use float32 buffers and assume VART already returned real values.
OUTPUT_FLOAT32_ALREADY_DEQUANT = False

_GRAPH = None  # keep graph alive; avoids Python GC issue


def log(msg: str):
    print(msg, flush=True)


# -------------------------
# Math helpers
# -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def xywh2xyxy(x):
    y = np.empty_like(x, dtype=np.float32)
    y[:, 0] = x[:, 0] - x[:, 2] / 2.0
    y[:, 1] = x[:, 1] - x[:, 3] / 2.0
    y[:, 2] = x[:, 0] + x[:, 2] / 2.0
    y[:, 3] = x[:, 1] + x[:, 3] / 2.0
    return y


def dfl(box_flat):
    """DFL decode: [N,64] -> [N,4] distances l,t,r,b."""
    n = box_flat.shape[0]
    b = box_flat.reshape(n, 4, REG_MAX)
    proj = np.arange(REG_MAX, dtype=np.float32)
    e = np.exp(b - b.max(axis=-1, keepdims=True))
    p = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-9)
    return (p * proj).sum(axis=-1)


# -------------------------
# VART / XIR
# -------------------------
def build_runner(xmodel_path: str):
    global _GRAPH
    _GRAPH = xir.Graph.deserialize(xmodel_path)
    root = _GRAPH.get_root_subgraph()
    children = root.toposort_child_subgraph()
    dpu_sg = None
    for sg in children:
        if sg.has_attr("device") and sg.get_attr("device").upper() == "DPU":
            dpu_sg = sg
            break
    if dpu_sg is None:
        raise RuntimeError("Cannot find DPU subgraph in xmodel")
    return vart.Runner.create_runner(dpu_sg, "run")


def tensor_fix(t):
    return int(t.get_attr("fix_point")) if t.has_attr("fix_point") else 0


def dequant(arr, fp):
    x = arr.astype(np.float32)
    if OUTPUT_FLOAT32_ALREADY_DEQUANT:
        return x
    return x / (2 ** fp)


def get_layout_from_dims(dims):
    # Returns 'NHWC' or 'NCHW'.
    if len(dims) != 4:
        return "UNKNOWN"
    if dims[1] in (1, 3, 32, 37, 64) and dims[2] == IMG_SIZE and dims[3] == IMG_SIZE:
        return "NCHW"
    if dims[3] in (1, 3, 32, 37, 64):
        return "NHWC"
    # common input [1,3,640,640]
    if dims[1] == 3:
        return "NCHW"
    return "NHWC"


def output_to_nhwc(arr):
    """Convert 4D output to NHWC if it looks NCHW."""
    if arr.ndim != 4:
        return arr
    # [1,C,H,W] -> [1,H,W,C]
    if arr.shape[1] in (1, 32, 64) and arr.shape[-1] not in (1, 32, 64):
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def proto_to_hwc(proto):
    """Accept [1,H,W,32] or [1,32,H,W] or [H,W,32] or [32,H,W]. Return [H,W,32]."""
    p = np.asarray(proto)
    if p.ndim == 4:
        p = p[0]
    if p.ndim != 3:
        raise RuntimeError(f"Bad proto shape: {proto.shape}")
    if p.shape[0] == NM and p.shape[-1] != NM:  # [32,H,W]
        p = np.transpose(p, (1, 2, 0))
    return p.astype(np.float32)


# -------------------------
# Preprocess / letterbox
# -------------------------
def letterbox(img_bgr, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def preprocess(img_bgr, input_tensor):
    dims = list(input_tensor.dims)
    fp = tensor_fix(input_tensor)
    layout = get_layout_from_dims(dims)

    canvas, scale, dw, dh = letterbox(img_bgr, IMG_SIZE)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if layout == "NCHW":
        x = np.transpose(rgb, (2, 0, 1))[None]  # [1,3,640,640]
    else:
        x = rgb[None]                           # [1,640,640,3]

    xq = (x * (2 ** fp)).clip(-128, 127).astype(np.int8)
    return np.ascontiguousarray(xq), scale, dw, dh, layout, fp


# -------------------------
# Decode raw split outputs: 10 tensors
# -------------------------
def map_split_outputs(out_tensors):
    """Map raw split outputs by shape. Returns dict key -> output index."""
    tmap = {}
    for i, t in enumerate(out_tensors):
        dims = list(t.dims)
        if len(dims) != 4:
            continue
        # Normalize dims to NHWC-like interpretation.
        if dims[1] in (1, NM, 4 * REG_MAX) and dims[3] not in (1, NM, 4 * REG_MAX):
            # NCHW
            H, W, C = dims[2], dims[3], dims[1]
        else:
            # NHWC
            H, W, C = dims[1], dims[2], dims[3]
        if C == NM and H == 160:
            tmap["proto"] = i
        elif C == 4 * REG_MAX and H in (80, 40, 20):
            tmap[f"box{IMG_SIZE // H}"] = i
        elif C == NC and H in (80, 40, 20):
            tmap[f"cls{IMG_SIZE // H}"] = i
        elif C == NM and H in (80, 40, 20):
            tmap[f"msk{IMG_SIZE // H}"] = i

    need = ["proto"] + [f"{r}{s}" for s in STRIDES for r in ("box", "cls", "msk")]
    return tmap if all(k in tmap for k in need) else None


def decode_raw_split(outputs, out_tensors, tmap):
    all_boxes, all_scores, all_coeffs = [], [], []

    def get(key):
        idx = tmap[key]
        fp = tensor_fix(out_tensors[idx])
        arr = dequant(outputs[idx], fp)
        if arr.ndim == 4:
            arr = output_to_nhwc(arr)[0]
        return arr

    for stride in STRIDES:
        box_f = get(f"box{stride}")  # [H,W,64]
        cls_f = get(f"cls{stride}")  # [H,W,1]
        msk_f = get(f"msk{stride}")  # [H,W,32]
        H, W = box_f.shape[:2]

        scores_hw = sigmoid(cls_f.reshape(-1))
        keep = scores_hw > CONF_THRESH
        if keep.sum() == 0:
            continue

        gy, gx = np.mgrid[0:H, 0:W]
        cx = (gx.ravel() + 0.5)[keep] * stride
        cy = (gy.ravel() + 0.5)[keep] * stride
        ltrb = dfl(box_f.reshape(-1, 4 * REG_MAX)[keep]) * stride

        x1 = cx - ltrb[:, 0]
        y1 = cy - ltrb[:, 1]
        x2 = cx + ltrb[:, 2]
        y2 = cy + ltrb[:, 3]

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        scores = scores_hw[keep].astype(np.float32)
        coeffs = msk_f.reshape(-1, NM)[keep].astype(np.float32)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_coeffs.append(coeffs)

    if len(all_boxes) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0, NM), np.float32),
                None)

    proto_idx = tmap["proto"]
    proto = dequant(outputs[proto_idx], tensor_fix(out_tensors[proto_idx]))
    proto = proto_to_hwc(proto)

    return (np.concatenate(all_boxes, axis=0),
            np.concatenate(all_scores, axis=0),
            np.concatenate(all_coeffs, axis=0),
            proto)


# -------------------------
# Decode Ultralytics-style outputs: pred + proto
# -------------------------
def find_ultralytics_outputs(outputs):
    pred_i, proto_i = None, None
    for i, o in enumerate(outputs):
        s = tuple(o.shape)
        if o.ndim == 3:
            # [1,37,8400] or [1,8400,37]
            if (s[1] == 4 + NC + NM) or (s[2] == 4 + NC + NM):
                pred_i = i
        elif o.ndim == 4:
            # proto [1,32,160,160] or [1,160,160,32]
            if (s[1] == NM and s[2] == 160) or (s[-1] == NM and s[1] == 160):
                proto_i = i
    return pred_i, proto_i


def decode_ultralytics(outputs, out_tensors, pred_i, proto_i):
    pred = dequant(outputs[pred_i], tensor_fix(out_tensors[pred_i]))
    pred = np.asarray(pred)
    p = pred[0]
    C = 4 + NC + NM
    if p.shape[0] == C:       # [C,N]
        p = p.T               # [N,C]
    elif p.shape[1] == C:
        pass                  # [N,C]
    else:
        raise RuntimeError(f"Bad pred shape: {pred.shape}")

    boxes_xywh = p[:, :4].astype(np.float32)
    scores = p[:, 4:4 + NC].reshape(-1).astype(np.float32)
    # Some raw outputs may be logits; Ultralytics export usually already has probabilities.
    if scores.max() > 1.0 or scores.min() < 0.0:
        scores = sigmoid(scores)
    coeffs = p[:, 4 + NC:4 + NC + NM].astype(np.float32)

    keep = scores > CONF_THRESH
    boxes = xywh2xyxy(boxes_xywh[keep])
    scores = scores[keep]
    coeffs = coeffs[keep]

    proto = dequant(outputs[proto_i], tensor_fix(out_tensors[proto_i]))
    proto = proto_to_hwc(proto)
    return boxes, scores, coeffs, proto


# -------------------------
# NMS / masks / row extraction
# -------------------------
def box_nms(boxes, scores):
    if len(boxes) == 0:
        return []
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, IMG_SIZE - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, IMG_SIZE - 1)
    w = b[:, 2] - b[:, 0]
    h = b[:, 3] - b[:, 1]
    valid = (w > 2) & (h > 2) & (scores >= CONF_THRESH)
    if valid.sum() == 0:
        return []

    orig = np.where(valid)[0]
    b = b[valid]
    s = scores[valid]
    order = np.argsort(-s)
    if len(order) > 500:
        order = order[:500]
    b = b[order]
    s = s[order]
    orig = orig[order]

    xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), CONF_THRESH, BOX_NMS_IOU)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)[:MAX_CANDIDATES]
    return orig[idx].tolist()


def build_proto_to_image_affine(ph, pw, scale, dw, dh):
    """
    Gộp 2 bước resize (proto->640, 640->ảnh gốc) thành 1 ma trận affine duy nhất.
    Lý do: mỗi lần cv2.resize với INTER_LINEAR làm nhòe biên một chút.
    2 lần liên tiếp khiến contour "phồng" ra so với object thật -> mask
    không bám sát. Gộp lại còn 1 phép nội suy duy nhất, biên sắc nét hơn.

    Mapping:  proto(u,v) -> 640-space (u*s, v*s) -> ảnh gốc ((x640-dw)/scale, ...)
              với s = IMG_SIZE / pw  (thường = 4, vì 640/160)
    """
    s = IMG_SIZE / pw
    a = s / scale
    M = np.array([
        [a, 0, -dw / scale],
        [0, a, -dh / scale],
    ], dtype=np.float32)
    return M


def instances_from_masks(boxes_640, scores, coeffs, proto_hwc,
                         img_h, img_w, scale, dw, dh):
    """
    Trả về list instance ĐỘC LẬP — giống cách Ultralytics/ONNX trên PC làm:
    mỗi box sau NMS có mask + score riêng, KHÔNG gộp vào nhau.

    Khác với make_union_mask cũ: không OR các mask lại thành 1 ảnh lớn,
    nên 2 hàng cây ở gần/chạm nhau vẫn được giữ là 2 instance riêng,
    đúng confidence score gốc thay vì chỉ đánh số thứ tự.
    """
    if len(boxes_640) == 0:
        return []

    ph, pw, pc = proto_hwc.shape
    assert pc == NM, f"Proto channel mismatch: {proto_hwc.shape}"
    proto_flat = proto_hwc.reshape(-1, NM).T  # [32, ph*pw]

    M = build_proto_to_image_affine(ph, pw, scale, dw, dh)
    # Kernel rất nhỏ chỉ để xóa nhiễu lốm đốm DO QUANTIZE bên trong
    # 1 mask — không liên quan/lan sang mask của instance khác.
    open_k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    instances = []
    for box, score, coeff in zip(boxes_640, scores, coeffs):
        raw = sigmoid(coeff @ proto_flat).reshape(ph, pw)

        bx1 = int(np.clip(box[0] * pw / IMG_SIZE, 0, pw - 1))
        by1 = int(np.clip(box[1] * ph / IMG_SIZE, 0, ph - 1))
        bx2 = int(np.clip(box[2] * pw / IMG_SIZE, 0, pw))
        by2 = int(np.clip(box[3] * ph / IMG_SIZE, 0, ph))

        crop = np.zeros_like(raw, dtype=np.float32)
        if bx2 > bx1 and by2 > by1:
            crop[by1:by2, bx1:bx2] = raw[by1:by2, bx1:bx2]

        mfull = cv2.warpAffine(
            crop, M, (img_w, img_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        mbin = (mfull > MASK_THRESH).astype(np.uint8)
        mbin = cv2.morphologyEx(mbin, cv2.MORPH_OPEN, open_k_small)

        if mbin.sum() < MIN_COMPONENT_AREA:
            continue

        ys, xs = np.where(mbin > 0)
        x1o, x2o = int(xs.min()), int(xs.max())
        y1o, y2o = int(ys.min()), int(ys.max())

        instances.append({
            "box": (x1o, y1o, x2o, y2o),
            "score": float(score),
            "mask": mbin,
        })

    return instances


def draw_instances(img_bgr, instances):
    """Vẽ từng instance riêng — bbox + contour + score thật, kiểu PC/Ultralytics."""
    out = img_bgr.copy()

    overlay = np.zeros_like(out)
    for inst in instances:
        overlay[inst["mask"] > 0] = COLOR_BGR
    out = cv2.addWeighted(out, 1.0, overlay, ALPHA, 0)

    for inst in instances:
        x1, y1, x2, y2 = inst["box"]
        score = inst["score"]
        mask = inst["mask"]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_BGR, 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_BGR, 2)

        label = f"crop_row {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        yy = max(y1, th + 8)
        cv2.rectangle(out, (x1, yy - th - 7), (x1 + tw + 5, yy), COLOR_BGR, -1)
        cv2.putText(out, label, (x1 + 2, yy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        # Fit line hướng hàng — tùy chọn, giữ lại từ bản trước
        pts = np.column_stack(np.where(mask > 0))
        if len(pts) > 20:
            xy = pts[:, ::-1].astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
            if abs(vx) > 1e-6:
                lx1, lx2 = x1, x2
                ly1 = int(y0 + (lx1 - x0) * vy / vx)
                ly2 = int(y0 + (lx2 - x0) * vy / vx)
                cv2.line(out, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2)


    return out


# -------------------------
# Main inference
# -------------------------
def run(xmodel_path, image_path, output_path):
    if not os.path.exists(xmodel_path):
        raise FileNotFoundError(f"xmodel not found: {xmodel_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"image not found: {image_path}")

    runner = build_runner(xmodel_path)
    in_tensors = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()

    log(f"XMODEL: {xmodel_path}")
    log(f"Input tensors: {len(in_tensors)}")
    for i, t in enumerate(in_tensors):
        log(f"  IN[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
    log(f"Output tensors: {len(out_tensors)}")
    for i, t in enumerate(out_tensors):
        log(f"  OUT[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
    log(f"Output mode: {'float32-auto-dequant' if OUTPUT_FLOAT32_ALREADY_DEQUANT else 'int8-manual-dequant'}")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    inp, scale, dw, dh, layout, fp_in = preprocess(img, in_tensors[0])
    log(f"Preprocess: layout={layout} scale={scale:.6f} dw={dw} dh={dh} input_fp={fp_in:+d}")

    out_dtype = np.float32 if OUTPUT_FLOAT32_ALREADY_DEQUANT else np.int8
    outputs = [np.zeros(list(t.dims), dtype=out_dtype) for t in out_tensors]

    t0 = time.time()
    job = runner.execute_async([inp], outputs)
    runner.wait(job)
    dpu_ms = (time.time() - t0) * 1000.0
    log(f"DPU: {dpu_ms:.1f} ms")

    t1 = time.time()
    split_map = map_split_outputs(out_tensors)
    if split_map is not None:
        log("Decode mode: raw split outputs")
        boxes, scores, coeffs, proto = decode_raw_split(outputs, out_tensors, split_map)
    else:
        pred_i, proto_i = find_ultralytics_outputs(outputs)
        if pred_i is None or proto_i is None:
            raise RuntimeError("Cannot recognize xmodel outputs. Need raw split 10 outputs or Ultralytics pred+proto outputs.")
        log("Decode mode: Ultralytics pred+proto outputs")
        boxes, scores, coeffs, proto = decode_ultralytics(outputs, out_tensors, pred_i, proto_i)

    log(f"Candidates after conf: {len(boxes)}")
    if len(boxes) == 0:
        cv2.imwrite(output_path, img)
        log(f"No candidates. Saved original image -> {output_path}")
        return

    # score stats
    log(f"Score: min={scores.min():.4f} max={scores.max():.4f} mean={scores.mean():.4f} >0.9={(scores>0.9).sum()} >0.99={(scores>0.99).sum()}")

    keep = box_nms(boxes, scores)
    log(f"After box NMS: {len(keep)}")
    if len(keep) == 0:
        cv2.imwrite(output_path, img)
        log(f"No boxes after NMS. Saved original image -> {output_path}")
        return

    boxes_f = boxes[keep]
    scores_f = scores[keep]
    coeffs_f = coeffs[keep]

    instances = instances_from_masks(boxes_f, scores_f, coeffs_f, proto,
                                     img_h, img_w, scale, dw, dh)
    arm_ms = (time.time() - t1) * 1000.0
    log(f"Instances kept: {len(instances)}")
    log(f"ARM postprocess: {arm_ms:.1f} ms")
    log(f"Total: {dpu_ms + arm_ms:.1f} ms ({1000.0 / max(dpu_ms + arm_ms, 1e-6):.2f} FPS)")

    if len(instances) == 0:
        cv2.imwrite(output_path, img)
        log(f"No instances after mask filter. Saved original image -> {output_path}")
        return

    out = draw_instances(img, instances)
    cv2.imwrite(output_path, out)
    log(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", default="field.jpg", help="input image")
    parser.add_argument("output", nargs="?", default="result.jpg", help="output image")
    parser.add_argument("--xmodel", default="croprow_yolov8.xmodel", help="xmodel path")
    args = parser.parse_args()
    run(args.xmodel, args.image, args.output)
