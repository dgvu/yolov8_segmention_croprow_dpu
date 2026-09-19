#!/usr/bin/env python3
# infer_video_sep.py - ZCU104 YOLOv8-seg video inference with better crop-row separation.
# Key idea: keep per-instance masks separate instead of merging everything into one union mask.

import argparse
import os
import time
from typing import Dict, List

import cv2
import numpy as np
import xir
import vart

IMG_SIZE = 640
NC = 1
NM = 32
REG_MAX = 16
STRIDES = [8, 16, 32]

# Balanced defaults: preserve recall better than tune1, but reduce merged segments.
CONF_THRESH = 0.30
BOX_NMS_IOU = 0.55
MAX_CANDIDATES = 25
MASK_THRESH = 0.55
MASK_IOU_THRESH = 0.25
MIN_MASK_AREA = 1800
MAX_MASK_AREA_RATIO = 0.35
MIN_ELONGATION = 1.35
MAX_FILL_RATIO = 0.80
MAX_FINAL_INSTANCES = 12
MASK_OPEN_KERNEL = (3, 3)
MASK_ERODE_KERNEL = (3, 3)
ALPHA = 0.38
COLOR_BGR = (0, 220, 80)
OUTPUT_FLOAT32_ALREADY_DEQUANT = False
_GRAPH = None


def log(msg):
    print(msg, flush=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def dfl(box_flat):
    n = box_flat.shape[0]
    b = box_flat.reshape(n, 4, REG_MAX)
    proj = np.arange(REG_MAX, dtype=np.float32)
    e = np.exp(b - b.max(axis=-1, keepdims=True))
    p = e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-9)
    return (p * proj).sum(axis=-1)


def tensor_fix(t):
    return int(t.get_attr("fix_point")) if t.has_attr("fix_point") else 0


def dequant(arr, fp):
    x = arr.astype(np.float32)
    if OUTPUT_FLOAT32_ALREADY_DEQUANT:
        return x
    return x / (2 ** fp)


def find_dpu_subgraph(sg):
    if sg.has_attr("device"):
        dev = sg.get_attr("device")
        if isinstance(dev, str) and dev.upper() == "DPU":
            return sg
    for child in sg.toposort_child_subgraph():
        found = find_dpu_subgraph(child)
        if found is not None:
            return found
    return None


def build_runner(xmodel_path):
    global _GRAPH
    _GRAPH = xir.Graph.deserialize(xmodel_path)
    root = _GRAPH.get_root_subgraph()
    dpu = find_dpu_subgraph(root)
    if dpu is None:
        raise RuntimeError("Cannot find DPU subgraph in xmodel")
    return vart.Runner.create_runner(dpu, "run")


def output_to_nhwc(arr):
    if arr.ndim != 4:
        return arr
    if arr.shape[1] in (1, 32, 64) and arr.shape[-1] not in (1, 32, 64):
        return np.transpose(arr, (0, 2, 3, 1))
    return arr


def proto_to_hwc(proto):
    p = np.asarray(proto)
    if p.ndim == 4:
        p = p[0]
    if p.ndim != 3:
        raise RuntimeError(f"Bad proto shape: {proto.shape}")
    if p.shape[0] == NM and p.shape[-1] != NM:
        p = np.transpose(p, (1, 2, 0))
    return p.astype(np.float32)


def get_layout_from_dims(dims):
    if len(dims) != 4:
        return "UNKNOWN"
    if dims[1] == 3:
        return "NCHW"
    return "NHWC"


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
        x = np.transpose(rgb, (2, 0, 1))[None]
    else:
        x = rgb[None]
    xq = (x * (2 ** fp)).clip(-128, 127).astype(np.int8)
    return np.ascontiguousarray(xq), scale, dw, dh


def map_split_outputs(out_tensors):
    tmap: Dict[str, int] = {}
    for i, t in enumerate(out_tensors):
        dims = list(t.dims)
        if len(dims) != 4:
            continue
        if dims[1] in (1, NM, 4 * REG_MAX) and dims[3] not in (1, NM, 4 * REG_MAX):
            H, W, C = dims[2], dims[3], dims[1]
        else:
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
    miss = [k for k in need if k not in tmap]
    if miss:
        raise RuntimeError(f"Missing output tensors: {miss}. Got map={tmap}")
    return tmap


def decode_raw_split(outputs, out_tensors, tmap):
    all_boxes, all_scores, all_coeffs = [], [], []

    def get(key):
        idx = tmap[key]
        arr = dequant(outputs[idx], tensor_fix(out_tensors[idx]))
        if arr.ndim == 4:
            arr = output_to_nhwc(arr)[0]
        return arr

    for stride in STRIDES:
        box_f = get(f"box{stride}")
        cls_f = get(f"cls{stride}")
        msk_f = get(f"msk{stride}")
        H, W = box_f.shape[:2]
        scores_hw = sigmoid(cls_f.reshape(-1))
        keep = scores_hw > CONF_THRESH
        if keep.sum() == 0:
            continue
        gy, gx = np.mgrid[0:H, 0:W]
        cx = (gx.ravel() + 0.5)[keep] * stride
        cy = (gy.ravel() + 0.5)[keep] * stride
        ltrb = dfl(box_f.reshape(-1, 4 * REG_MAX)[keep]) * stride
        boxes = np.stack([cx - ltrb[:, 0], cy - ltrb[:, 1], cx + ltrb[:, 2], cy + ltrb[:, 3]], axis=1).astype(np.float32)
        scores = scores_hw[keep].astype(np.float32)
        coeffs = msk_f.reshape(-1, NM)[keep].astype(np.float32)
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_coeffs.append(coeffs)

    if len(all_boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0, NM), np.float32), None

    proto_idx = tmap["proto"]
    proto = dequant(outputs[proto_idx], tensor_fix(out_tensors[proto_idx]))
    proto = proto_to_hwc(proto)
    return np.concatenate(all_boxes, 0), np.concatenate(all_scores, 0), np.concatenate(all_coeffs, 0), proto


def box_nms(boxes, scores):
    if len(boxes) == 0:
        return []
    b = boxes.copy().astype(np.float32)
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, IMG_SIZE - 1)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, IMG_SIZE - 1)
    valid = ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2) & (scores >= CONF_THRESH)
    if valid.sum() == 0:
        return []
    orig = np.where(valid)[0]
    b = b[valid]
    s = scores[valid]
    order = np.argsort(-s)
    if len(order) > 600:
        order = order[:600]
    b = b[order]
    s = s[order]
    orig = orig[order]
    xywh = np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), s.tolist(), CONF_THRESH, BOX_NMS_IOU)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1)[:MAX_CANDIDATES]
    return orig[idx].tolist()


def largest_component(mask):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    best_i = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == best_i).astype(np.uint8)


def refine_instance_mask(mask):
    mask = (mask > 0).astype(np.uint8)
    if mask.sum() == 0:
        return mask
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, MASK_OPEN_KERNEL)
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, MASK_ERODE_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.erode(mask, erode_k, iterations=1)
    mask = largest_component(mask)
    return mask


def make_instance_masks(boxes_640, scores, coeffs, proto_hwc, img_h, img_w, scale, dw, dh):
    if len(boxes_640) == 0:
        return []
    ph, pw, _ = proto_hwc.shape
    proto_flat = proto_hwc.reshape(-1, NM).T
    nh_lb = int(round(img_h * scale))
    nw_lb = int(round(img_w * scale))
    img_area = img_h * img_w
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
        m640 = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        mletter = m640[dh:dh + nh_lb, dw:dw + nw_lb]
        if mletter.size == 0:
            continue
        mfull = cv2.resize(mletter, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        mask = refine_instance_mask((mfull > MASK_THRESH).astype(np.uint8))
        area = int(mask.sum())
        if area < MIN_MASK_AREA:
            continue
        if area > MAX_MASK_AREA_RATIO * img_area:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        elong = max(w, h) / max(1, min(w, h))
        if elong < MIN_ELONGATION:
            continue
        fill_ratio = float(area) / max(1, w * h)
        if fill_ratio > MAX_FILL_RATIO:
            continue
        instances.append({
            "mask": mask,
            "score": float(score),
            "bbox": (x1, y1, w, h),
            "area": area,
            "elong": elong,
        })
    return instances


def mask_iou(m1, m2):
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    if inter == 0:
        return 0.0
    uni = np.logical_or(m1 > 0, m2 > 0).sum()
    return float(inter) / max(1, uni)


def mask_iou_nms(instances):
    if not instances:
        return []
    order = sorted(range(len(instances)), key=lambda i: (-instances[i]["score"], -instances[i]["area"]))
    keep = []
    for i in order:
        ok = True
        for j in keep:
            if mask_iou(instances[i]["mask"], instances[j]["mask"]) > MASK_IOU_THRESH:
                ok = False
                break
        if ok:
            keep.append(i)
        if len(keep) >= MAX_FINAL_INSTANCES:
            break
    return [instances[i] for i in keep]


def draw_instances(img_bgr, instances):
    out = img_bgr.copy()
    for idx, inst in enumerate(instances):
        mask = inst["mask"]
        x, y, w, h = inst["bbox"]
        overlay = np.zeros_like(out)
        overlay[mask > 0] = COLOR_BGR
        out = cv2.addWeighted(out, 1.0, overlay, ALPHA, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, COLOR_BGR, 2)
        cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_BGR, 2)
        label = f"crop_row {idx + 1}"
        yy = max(y, 20)
        cv2.rectangle(out, (x, yy - 18), (x + 95, yy + 2), COLOR_BGR, -1)
        cv2.putText(out, label, (x + 2, yy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        pts = np.column_stack(np.where(mask > 0))
        if len(pts) > 20:
            xy = pts[:, ::-1].astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(xy, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
            if abs(vx) > 1e-6:
                x1l, x2l = x, x + w
                y1l = int(y0 + (x1l - x0) * vy / vx)
                y2l = int(y0 + (x2l - x0) * vy / vx)
                cv2.line(out, (x1l, y1l), (x2l, y2l), (0, 0, 255), 2)
    return out


class VideoInferencer:
    def __init__(self, xmodel_path, verbose=True):
        self.runner = build_runner(xmodel_path)
        self.in_tensors = self.runner.get_input_tensors()
        self.out_tensors = self.runner.get_output_tensors()
        self.out_dtype = np.float32 if OUTPUT_FLOAT32_ALREADY_DEQUANT else np.int8
        self.tmap = map_split_outputs(self.out_tensors)
        if verbose:
            log(f"XMODEL: {xmodel_path}")
            for i, t in enumerate(self.in_tensors):
                log(f"  IN[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
            for i, t in enumerate(self.out_tensors):
                log(f"  OUT[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
            log("Decode mode: raw split outputs")
            log(f"Params: conf={CONF_THRESH} box_iou={BOX_NMS_IOU} max_candidates={MAX_CANDIDATES} mask={MASK_THRESH} mask_iou={MASK_IOU_THRESH}")

    def process_frame(self, frame):
        img_h, img_w = frame.shape[:2]
        inp, scale, dw, dh = preprocess(frame, self.in_tensors[0])
        outputs = [np.zeros(list(t.dims), dtype=self.out_dtype) for t in self.out_tensors]
        t0 = time.time()
        job = self.runner.execute_async([inp], outputs)
        self.runner.wait(job)
        dpu_ms = (time.time() - t0) * 1000.0
        t1 = time.time()
        boxes, scores, coeffs, proto = decode_raw_split(outputs, self.out_tensors, self.tmap)
        keep_box = box_nms(boxes, scores)
        if len(keep_box) == 0:
            return frame.copy(), {"dpu_ms": dpu_ms, "arm_ms": (time.time() - t1) * 1000.0, "cand": len(boxes), "nms": 0, "inst": 0, "final": 0}
        instances = make_instance_masks(boxes[keep_box], scores[keep_box], coeffs[keep_box], proto, img_h, img_w, scale, dw, dh)
        final_instances = mask_iou_nms(instances)
        out = draw_instances(frame, final_instances)
        arm_ms = (time.time() - t1) * 1000.0
        return out, {"dpu_ms": dpu_ms, "arm_ms": arm_ms, "cand": len(boxes), "nms": len(keep_box), "inst": len(instances), "final": len(final_instances)}


def fourcc_from_name(name):
    name = (name or "MJPG").upper()
    if len(name) != 4:
        name = "MJPG"
    return cv2.VideoWriter_fourcc(*name)


def run_video(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps <= 0 or np.isnan(in_fps):
        in_fps = 20.0
    out_fps = args.output_fps if args.output_fps else in_fps / max(args.frame_step, 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"VIDEO: {args.video} {w}x{h} fps={in_fps:.2f} frames={total}")
    log(f"OUTPUT: {args.output} fps={out_fps:.2f} codec={args.codec}")
    if args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
    inferencer = VideoInferencer(args.xmodel, verbose=True)
    writer = None
    read_i = 0
    write_i = 0
    sum_dpu = sum_arm = sum_total = 0.0
    t_all = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_i % max(args.frame_step, 1) != 0:
            read_i += 1
            continue
        if args.max_frames > 0 and write_i >= args.max_frames:
            break
        t0 = time.time()
        out, info = inferencer.process_frame(frame)
        total_ms = (time.time() - t0) * 1000.0
        sum_dpu += info["dpu_ms"]
        sum_arm += info["arm_ms"]
        sum_total += total_ms
        if writer is None:
            oh, ow = out.shape[:2]
            writer = cv2.VideoWriter(args.output, fourcc_from_name(args.codec), out_fps, (ow, oh))
            if not writer.isOpened():
                raise RuntimeError("Cannot open VideoWriter. Try output .avi with --codec MJPG or --codec XVID")
        writer.write(out)
        if args.frames_dir:
            cv2.imwrite(os.path.join(args.frames_dir, f"frame_{write_i:06d}.jpg"), out)
        if write_i % args.log_every == 0:
            log(f"frame {write_i:05d} src={read_i:05d} DPU={info['dpu_ms']:.1f}ms ARM={info['arm_ms']:.1f}ms Total={total_ms:.1f}ms cand={info['cand']} box_nms={info['nms']} inst={info['inst']} final={info['final']}")
        read_i += 1
        write_i += 1
    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t_all
    if write_i > 0:
        log("---------------- SUMMARY ----------------")
        log(f"Processed frames: {write_i}")
        log(f"Elapsed: {elapsed:.2f}s")
        log(f"Avg DPU: {sum_dpu/write_i:.1f}ms")
        log(f"Avg ARM: {sum_arm/write_i:.1f}ms")
        log(f"Avg total: {sum_total/write_i:.1f}ms")
        log(f"Processing FPS: {write_i/max(elapsed,1e-9):.2f}")
        log(f"Saved -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="input video path")
    parser.add_argument("output", nargs="?", default="result_video_sep.avi", help="output video")
    parser.add_argument("--xmodel", default="croprow_yolov8.xmodel")
    parser.add_argument("--frames_dir", default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--codec", default="MJPG")
    parser.add_argument("--output_fps", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max_candidates", type=int, default=None)
    parser.add_argument("--mask_thresh", type=float, default=None)
    parser.add_argument("--mask_iou", type=float, default=None)
    args = parser.parse_args()
    if args.conf is not None:
        CONF_THRESH = args.conf
    if args.iou is not None:
        BOX_NMS_IOU = args.iou
    if args.max_candidates is not None:
        MAX_CANDIDATES = args.max_candidates
    if args.mask_thresh is not None:
        MASK_THRESH = args.mask_thresh
    if args.mask_iou is not None:
        MASK_IOU_THRESH = args.mask_iou
    run_video(args)
