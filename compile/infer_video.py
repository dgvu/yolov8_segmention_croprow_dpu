# infer.py — ZCU104  (fixed: graph GC + dequantize)
# Bug 1 fix: graph phải là global, không được để GC thu hồi
# Bug 2 fix: output là INT8 thô, phải dequant bằng fixpos

import sys
import time
import cv2
import numpy as np
import xir
import vart

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
XMODEL      = "croprow_yolov8.xmodel"
IMG_SIZE    = 640
NC          = 1      # crop_row
NM          = 32
REG_MAX     = 16     # box channel = 4×16 = 64
STRIDES     = [8, 16, 32]
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
MASK_THRESH = 0.5
COLOR_BGR   = (0, 220, 80)

# ── FIX BUG 1: giữ graph sống suốt vòng đời chương trình ──
_GRAPH = None   # global → không bị Python GC thu hồi


# ═══════════════════════════════════════════
#  PHẦN 1 — DPU setup
# ═══════════════════════════════════════════

def build_runner(xmodel_path):
    global _GRAPH
    _GRAPH = xir.Graph.deserialize(xmodel_path)   # global → không bị GC
    dpu    = next(
        s for s in _GRAPH.get_root_subgraph().toposort_child_subgraph()
        if s.has_attr("device") and s.get_attr("device").upper() == "DPU"
    )
    return vart.Runner.create_runner(dpu, "run")


def map_tensors(out_tensors):
    """
    Nhận dạng 10 output tensor theo shape NHWC.
    Trả về dict key → {"tensor": t, "fp": fixpos}
    """
    tmap = {}
    for t in out_tensors:
        dims   = list(t.dims)           # [1, H, W, C]
        H, C   = dims[1], dims[3]
        fp_val = t.get_attr("fix_point") if t.has_attr("fix_point") else 0
        stride = IMG_SIZE // H

        if   C == NM and H == 160:     tmap["proto"]        = (t, fp_val)
        elif C == 4 * REG_MAX:         tmap[f"box{stride}"] = (t, fp_val)
        elif C == NC:                  tmap[f"cls{stride}"] = (t, fp_val)
        elif C == NM:                  tmap[f"msk{stride}"] = (t, fp_val)
        else:
            print(f"[WARN] tensor không nhận dạng: dims={dims}")

    need = ["proto"] + [f"{r}{s}" for s in STRIDES
                                   for r in ["box", "cls", "msk"]]
    miss = [k for k in need if k not in tmap]
    if miss:
        raise RuntimeError(f"Thiếu tensor: {miss}")
    return tmap


# ═══════════════════════════════════════════
#  PHẦN 2 — Preprocess (INT8 input)
# ═══════════════════════════════════════════

def letterbox(img):
    h, w   = img.shape[:2]
    scale  = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    dw, dh = (IMG_SIZE - nw) // 2, (IMG_SIZE - nh) // 2
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, np.uint8)
    canvas[dh:dh+nh, dw:dw+nw] = cv2.resize(img, (nw, nh))
    return canvas, scale, dw, dh


def preprocess(img_bgr, fp_input):
    canvas, scale, dw, dh = letterbox(img_bgr)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Quantize thủ công theo fixpos của input tensor
    inp = (rgb * (2 ** fp_input)).clip(-128, 127).astype(np.int8)
    return np.ascontiguousarray(inp[np.newaxis]), scale, dw, dh


# ═══════════════════════════════════════════
#  PHẦN 3 — Decode (dequantize + DFL + sigmoid)
# ═══════════════════════════════════════════

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def dfl(box_flat):
    """[N, 64] → [N, 4]  ltrb distances"""
    n    = box_flat.shape[0]
    b    = box_flat.reshape(n, 4, REG_MAX)
    proj = np.arange(REG_MAX, dtype=np.float32)
    e    = np.exp(b - b.max(-1, keepdims=True))
    prob = e / e.sum(-1, keepdims=True)
    return (prob * proj).sum(-1)


def decode_scale(box_buf, fp_box,
                 cls_buf, fp_cls,
                 msk_buf, fp_msk, stride):
    """
    Nhận INT8-as-float32 buffers + fixpos của 1 stride.
    Dequantize → decode boxes/scores/coeffs.
    """
    H = box_buf.shape[1]

    # ── FIX BUG 2: dequantize = value / 2^fixpos ──────────
    box_f = box_buf[0].astype(np.float32) / (2 ** fp_box)   # [H,W,64]
    cls_f = cls_buf[0].astype(np.float32) / (2 ** fp_cls)   # [H,W,1]
    msk_f = msk_buf[0].astype(np.float32) / (2 ** fp_msk)   # [H,W,32]

    scores_hw = sigmoid(cls_f.reshape(-1))
    keep      = scores_hw > CONF_THRESH
    if keep.sum() == 0:
        z = np.zeros((0,), np.float32)
        return np.zeros((0,4), np.float32), z, np.zeros((0,NM), np.float32)

    gy, gx = np.mgrid[0:H, 0:H]
    cx = (gx.ravel() + 0.5)[keep] * stride
    cy = (gy.ravel() + 0.5)[keep] * stride

    ltrb = dfl(box_f.reshape(-1, 4*REG_MAX)[keep]) * stride
    x1 = cx - ltrb[:,0];  y1 = cy - ltrb[:,1]
    x2 = cx + ltrb[:,2];  y2 = cy + ltrb[:,3]

    boxes  = np.stack([x1,y1,x2,y2], 1)
    scores = scores_hw[keep]
    coeffs = msk_f.reshape(-1, NM)[keep]
    return boxes, scores, coeffs


# ═══════════════════════════════════════════
#  PHẦN 4 — NMS
# ═══════════════════════════════════════════

def run_nms(boxes, scores):
    if len(boxes) == 0:
        return []
    idx = cv2.dnn.NMSBoxes(
        np.clip(boxes, 0, IMG_SIZE).tolist(),
        scores.tolist(), CONF_THRESH, IOU_THRESH
    )
    return idx.flatten().tolist() if len(idx) else []


# ═══════════════════════════════════════════
#  PHẦN 5 — Mask assembly
# ═══════════════════════════════════════════

def make_masks(coeffs, proto_buf, fp_proto,
               boxes_640, img_h, img_w, scale, dw, dh):
    # Dequantize proto
    proto  = proto_buf[0].astype(np.float32) / (2 ** fp_proto)  # [160,160,32]
    p_flat = proto.reshape(-1, NM).T                              # [32, 25600]
    s160   = 160.0 / IMG_SIZE

    out = []
    for coeff, box in zip(coeffs, boxes_640):
        raw  = sigmoid(coeff @ p_flat).reshape(160, 160)

        bx1 = int(np.clip(box[0] * s160, 0, 160))
        by1 = int(np.clip(box[1] * s160, 0, 160))
        bx2 = int(np.clip(box[2] * s160, 0, 160))
        by2 = int(np.clip(box[3] * s160, 0, 160))
        crop = np.zeros_like(raw)
        if bx2 > bx1 and by2 > by1:
            crop[by1:by2, bx1:bx2] = raw[by1:by2, bx1:bx2]

        m640  = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                           interpolation=cv2.INTER_LINEAR)
        nh_lb = int(img_h * scale)
        nw_lb = int(img_w * scale)
        mcrop = m640[dh:dh+nh_lb, dw:dw+nw_lb]
        mfull = cv2.resize(mcrop, (img_w, img_h),
                           interpolation=cv2.INTER_LINEAR)
        out.append((mfull > MASK_THRESH).astype(np.uint8))
    return out


# ═══════════════════════════════════════════
#  PHẦN 6 — Draw
# ═══════════════════════════════════════════

def draw(img_bgr, boxes_640, scores, masks,
         img_h, img_w, scale, dw, dh):
    out = img_bgr.copy()
    for box, score, mask in zip(boxes_640, scores, masks):
        x1 = int(np.clip((box[0]-dw)/scale, 0, img_w))
        y1 = int(np.clip((box[1]-dh)/scale, 0, img_h))
        x2 = int(np.clip((box[2]-dw)/scale, 0, img_w))
        y2 = int(np.clip((box[3]-dh)/scale, 0, img_h))

        overlay = np.zeros_like(out)
        overlay[mask == 1] = COLOR_BGR
        out = cv2.addWeighted(out, 1.0, overlay, 0.45, 0)

        cv2.rectangle(out, (x1,y1), (x2,y2), COLOR_BGR, 2)
        label       = f"crop_row {score:.2f}"
        (tw,th), _  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), COLOR_BGR, -1)
        cv2.putText(out, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    return out


# ═══════════════════════════════════════════
#  PHẦN 7 — Pipeline chính
# ═══════════════════════════════════════════

def run(image_path, output_path="result.jpg"):
    runner  = build_runner(XMODEL)          # _GRAPH giữ graph sống
    in_t    = runner.get_input_tensors()
    out_t   = runner.get_output_tensors()
    tmap    = map_tensors(out_t)
    fp_in   = in_t[0].get_attr("fix_point") if in_t[0].has_attr("fix_point") else 6

    print(f"Input : {list(in_t[0].dims)}  fp={fp_in}", flush=True)
    print("Outputs:", flush=True)
    for k in sorted(tmap):
        t, fp = tmap[k]
        print(f"  {k:8s} {list(t.dims)}  fp={fp:+d}  "
              f"scale={1/(2**fp):.4f}", flush=True)

    img_bgr = cv2.imread(image_path)
    assert img_bgr is not None
    img_h, img_w = img_bgr.shape[:2]

    inp, scale, dw, dh = preprocess(img_bgr, fp_in)

    # Alloc float32 output buffers (DPU ghi INT8 value vào float32 buffer)
    bufs     = [np.zeros(list(t.dims), dtype=np.float32) for t, _ in
                [tmap[k] for k in sorted(tmap)]]
    # Map theo tên tensor để lookup đúng buffer
    t_order  = [tmap[k][0] for k in sorted(tmap)]
    # Cần giữ đúng thứ tự mà runner.get_output_tensors() trả về
    bufs_ord = [np.zeros(list(t.dims), dtype=np.float32) for t in out_t]
    name2buf = {t.name: bufs_ord[i] for i, t in enumerate(out_t)}

    def get_buf(key):
        t, _ = tmap[key]
        return name2buf[t.name]

    def get_fp(key):
        return tmap[key][1]

    # ── DPU inference ────────────────────────
    t0  = time.time()
    job = runner.execute_async([inp], bufs_ord)
    runner.wait(job)
    ms_dpu = (time.time()-t0)*1000
    print(f"\nDPU  : {ms_dpu:.1f} ms", flush=True)

    # Debug: kiểm tra output range sau dequantize
    for k in ["cls8", "cls16", "cls32"]:
        buf = get_buf(k)
        fp  = get_fp(k)
        dq  = buf.astype(np.float32) / (2**fp)
        sc  = sigmoid(dq)
        print(f"  {k}: raw=[{buf.min():.0f},{buf.max():.0f}]  "
              f"dequant=[{dq.min():.2f},{dq.max():.2f}]  "
              f"sigmoid_max={sc.max():.3f}", flush=True)

    # ── ARM decode ───────────────────────────
    t1 = time.time()
    all_boxes, all_scores, all_coeffs = [], [], []
    for s in STRIDES:
        b, sc, co = decode_scale(
            get_buf(f"box{s}"), get_fp(f"box{s}"),
            get_buf(f"cls{s}"), get_fp(f"cls{s}"),
            get_buf(f"msk{s}"), get_fp(f"msk{s}"),
            stride=s
        )
        all_boxes.append(b)
        all_scores.append(sc)
        all_coeffs.append(co)

    boxes_all  = np.concatenate(all_boxes,  0)
    scores_all = np.concatenate(all_scores, 0)
    coeffs_all = np.concatenate(all_coeffs, 0)
    print(f"Candidates before NMS: {len(boxes_all)}", flush=True)

    keep = run_nms(boxes_all, scores_all)
    if not keep:
        print("Không detect được hàng cây nào — thử giảm CONF_THRESH")
        cv2.imwrite(output_path, img_bgr)
        return

    boxes_f  = boxes_all[keep]
    scores_f = scores_all[keep]
    coeffs_f = coeffs_all[keep]
    print(f"Detect: {len(boxes_f)} crop rows", flush=True)

    masks = make_masks(coeffs_f, get_buf("proto"), get_fp("proto"),
                       boxes_f, img_h, img_w, scale, dw, dh)

    ms_arm = (time.time()-t1)*1000
    print(f"ARM  : {ms_arm:.1f} ms")
    print(f"Total: {ms_dpu+ms_arm:.1f} ms  ({1000/(ms_dpu+ms_arm):.1f} FPS)")

    result = draw(img_bgr, boxes_f, scores_f, masks, img_h, img_w, scale, dw, dh)
    cv2.imwrite(output_path, result)
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    img_in  = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    img_out = sys.argv[2] if len(sys.argv) > 2 else "result.jpg"
    run(img_in, img_out)
# -------------------------
# Video inference - no imshow, no display required
# -------------------------
class VideoInferencer:
    def __init__(self, xmodel_path: str, verbose: bool = True):
        self.runner = build_runner(xmodel_path)
        self.in_tensors = self.runner.get_input_tensors()
        self.out_tensors = self.runner.get_output_tensors()
        self.out_dtype = np.float32 if OUTPUT_FLOAT32_ALREADY_DEQUANT else np.int8
        self.split_map = map_split_outputs(self.out_tensors)
        self.pred_i, self.proto_i = find_ultralytics_outputs([np.zeros(list(t.dims), dtype=self.out_dtype) for t in self.out_tensors])

        if verbose:
            log(f"XMODEL: {xmodel_path}")
            log(f"Input tensors: {len(self.in_tensors)}")
            for i, t in enumerate(self.in_tensors):
                log(f"  IN[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
            log(f"Output tensors: {len(self.out_tensors)}")
            for i, t in enumerate(self.out_tensors):
                log(f"  OUT[{i}] {list(t.dims)} fp={tensor_fix(t):+d}")
            if self.split_map is not None:
                log("Decode mode: raw split outputs")
            elif self.pred_i is not None and self.proto_i is not None:
                log("Decode mode: Ultralytics pred+proto outputs")
            else:
                raise RuntimeError("Cannot recognize xmodel outputs. Need raw split 10 outputs or Ultralytics pred+proto outputs.")
            log(f"Params: CONF={CONF_THRESH} NMS_IOU={BOX_NMS_IOU} MAX_CAND={MAX_CANDIDATES} MASK={MASK_THRESH}")

    def process_frame(self, img_bgr):
        img_h, img_w = img_bgr.shape[:2]
        inp, scale, dw, dh, layout, fp_in = preprocess(img_bgr, self.in_tensors[0])
        outputs = [np.zeros(list(t.dims), dtype=self.out_dtype) for t in self.out_tensors]

        t0 = time.time()
        job = self.runner.execute_async([inp], outputs)
        self.runner.wait(job)
        dpu_ms = (time.time() - t0) * 1000.0

        t1 = time.time()
        if self.split_map is not None:
            boxes, scores, coeffs, proto = decode_raw_split(outputs, self.out_tensors, self.split_map)
        else:
            boxes, scores, coeffs, proto = decode_ultralytics(outputs, self.out_tensors, self.pred_i, self.proto_i)

        if len(boxes) == 0:
            return img_bgr.copy(), {"dpu_ms": dpu_ms, "arm_ms": (time.time() - t1) * 1000.0, "cand": 0, "nms": 0, "masks": 0, "comps": 0}

        keep = box_nms(boxes, scores)
        if len(keep) == 0:
            return img_bgr.copy(), {"dpu_ms": dpu_ms, "arm_ms": (time.time() - t1) * 1000.0, "cand": len(boxes), "nms": 0, "masks": 0, "comps": 0}

        boxes_f = boxes[keep]
        scores_f = scores[keep]
        coeffs_f = coeffs[keep]

        union, inst_masks = make_union_mask(boxes_f, scores_f, coeffs_f, proto, img_h, img_w, scale, dw, dh)
        clean, comps = clean_union_and_components(union)
        out = draw_result(img_bgr, clean, comps)
        arm_ms = (time.time() - t1) * 1000.0

        return out, {
            "dpu_ms": dpu_ms,
            "arm_ms": arm_ms,
            "cand": int(len(boxes)),
            "nms": int(len(keep)),
            "masks": int(len(inst_masks)),
            "comps": int(len(comps)),
        }


def fourcc_from_name(name: str):
    name = (name or "MJPG").upper()
    if len(name) != 4:
        name = "MJPG"
    return cv2.VideoWriter_fourcc(*name)


def run_video(xmodel_path, video_path, output_path, frames_dir=None, frame_step=1, max_frames=-1, codec="MJPG", output_fps=None):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")
    if frame_step < 1:
        frame_step = 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if in_fps <= 0 or np.isnan(in_fps):
        in_fps = 20.0
    out_fps = float(output_fps) if output_fps else float(in_fps / frame_step)
    if out_fps <= 0:
        out_fps = 20.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"VIDEO: {video_path}")
    log(f"Input video: {width}x{height}, fps={in_fps:.2f}, frames={total_frames}")
    log(f"Output video: {output_path}, fps={out_fps:.2f}, codec={codec}, frame_step={frame_step}")

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
        log(f"Save processed frames to: {frames_dir}")

    inferencer = VideoInferencer(xmodel_path, verbose=True)

    writer = None
    read_idx = 0
    write_idx = 0
    sum_dpu = 0.0
    sum_arm = 0.0
    sum_total = 0.0
    t_all = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_idx % frame_step != 0:
            read_idx += 1
            continue
        if max_frames > 0 and write_idx >= max_frames:
            break

        t0 = time.time()
        out_frame, info = inferencer.process_frame(frame)
        total_ms = (time.time() - t0) * 1000.0
        sum_dpu += info["dpu_ms"]
        sum_arm += info["arm_ms"]
        sum_total += total_ms

        if writer is None:
            h, w = out_frame.shape[:2]
            # Use AVI+MJPG by default on ZCU104 because it is usually available without extra plugins.
            writer = cv2.VideoWriter(output_path, fourcc_from_name(codec), out_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter: {output_path}. Try --codec XVID and .avi output.")

        writer.write(out_frame)

        if frames_dir:
            cv2.imwrite(os.path.join(frames_dir, f"frame_{write_idx:06d}.jpg"), out_frame)

        if write_idx % 10 == 0:
            log(
                f"frame {write_idx:05d} src={read_idx:05d} "
                f"DPU={info['dpu_ms']:.1f}ms ARM={info['arm_ms']:.1f}ms Total={total_ms:.1f}ms "
                f"cand={info['cand']} nms={info['nms']} masks={info['masks']} comps={info['comps']}"
            )

        read_idx += 1
        write_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.time() - t_all
    if write_idx > 0:
        log("---------------- SUMMARY ----------------")
        log(f"Processed frames: {write_idx}")
        log(f"Elapsed wall time: {elapsed:.2f} s")
        log(f"Avg DPU: {sum_dpu / write_idx:.1f} ms")
        log(f"Avg ARM: {sum_arm / write_idx:.1f} ms")
        log(f"Avg total/frame: {sum_total / write_idx:.1f} ms")
        log(f"Processing FPS: {write_idx / max(elapsed, 1e-9):.2f}")
        log(f"Saved video -> {output_path}")
    else:
        log("No frames processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="input video path")
    parser.add_argument("output", nargs="?", default="result_video.avi", help="output video path, .avi recommended on ZCU104")
    parser.add_argument("--xmodel", default="croprow_yolov8.xmodel", help="compiled xmodel path")
    parser.add_argument("--frames_dir", default=None, help="optional folder to save processed frames as JPG")
    parser.add_argument("--frame_step", type=int, default=1, help="process every Nth frame; 1 = all frames")
    parser.add_argument("--max_frames", type=int, default=-1, help="limit number of processed frames for quick test")
    parser.add_argument("--codec", default="MJPG", help="fourcc codec, e.g. MJPG, XVID, mp4v")
    parser.add_argument("--output_fps", type=float, default=None, help="override output fps")
    args = parser.parse_args()
    run_video(args.xmodel, args.video, args.output, args.frames_dir, args.frame_step, args.max_frames, args.codec, args.output_fps)
