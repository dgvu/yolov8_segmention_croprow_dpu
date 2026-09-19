#!/usr/bin/env python3
# quant_croprow_raw.py
# DPU-friendly PyTorch quantization for YOLOv8-seg crop_row.
# Fixes common Vitis-AI export_xmodel failures:
#   - C2f chunk -> slicing, to avoid nndct_chunk multi-output op
#   - Segment head full decode -> raw outputs, to avoid meshgrid/arange/unbind/split
#   - optional SiLU -> LeakyReLU, to avoid unsupported/float SiLU op
# Outputs raw tensors for ARM postprocess:
#   proto, box8, cls8, msk8, box16, cls16, msk16, box32, cls32, msk32

import argparse
import glob
import os
import types

import cv2
import numpy as np
import torch
from pytorch_nndct.apis import torch_quantizer


def patch_c2f_no_chunk():
    """Monkey-patch Ultralytics C2f.forward to avoid .chunk() multi-output op."""
    try:
        from ultralytics.nn.modules import C2f
    except Exception:
        from ultralytics.nn.modules.block import C2f

    def forward_slice(self, x):
        y = self.cv1(x)
        # replace y.chunk(2, 1) with single-output slicing ops
        y0 = y[:, :self.c, :, :]
        y1 = y[:, self.c:, :, :]
        outs = [y0, y1]
        for m in self.m:
            outs.append(m(outs[-1]))
        return self.cv2(torch.cat(outs, 1))

    C2f.forward = forward_slice
    print("[PATCH] C2f.forward: chunk -> slicing")


def replace_silu_with_leakyrelu(module, negative_slope=0.1):
    """Replace SiLU activations recursively. This may slightly change accuracy; fine-tune if needed."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.SiLU):
            setattr(module, name, torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            count += 1
        else:
            count += replace_silu_with_leakyrelu(child, negative_slope)
    return count


def patch_segment_raw(model):
    """Patch YOLOv8 Segment head to return raw DPU-friendly outputs instead of decoded preds."""

    def raw_segment_forward(self, x):
        # x is list of feature maps [P3, P4, P5]
        proto = self.proto(x[0])
        outs = [proto]
        for i in range(self.nl):
            box = self.cv2[i](x[i])  # [B, 4*reg_max, H, W], usually 64 ch
            cls = self.cv3[i](x[i])  # [B, nc, H, W]
            msk = self.cv4[i](x[i])  # [B, nm, H, W]
            outs.extend([box, cls, msk])
        return tuple(outs)

    patched = 0
    for m in model.modules():
        if all(hasattr(m, a) for a in ["proto", "cv2", "cv3", "cv4", "nl"]):
            m.forward = types.MethodType(raw_segment_forward, m)
            patched += 1
    if patched == 0:
        raise RuntimeError("Cannot find YOLOv8 Segment head to patch")
    print(f"[PATCH] Segment.forward -> raw outputs, patched heads: {patched}")


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(path, imgsz):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")
    img = letterbox(img, imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0)


class CroprowRawWrapper(torch.nn.Module):
    def __init__(self, pt_path, replace_silu=True):
        super().__init__()
        patch_c2f_no_chunk()
        from ultralytics import YOLO
        self.model = YOLO(pt_path).model
        self.model.eval()
        patch_segment_raw(self.model)
        if replace_silu:
            n = replace_silu_with_leakyrelu(self.model, negative_slope=26/256)
            print(f"[PATCH] SiLU -> LeakyReLU: {n} modules")

    def forward(self, x):
        return self.model(x)


def print_output_shapes(model, imgsz):
    model.eval()
    x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        y = model(x)
    print("[CHECK] output type:", type(y), "len=", len(y) if isinstance(y, (tuple, list)) else "-")
    if isinstance(y, (tuple, list)):
        for i, t in enumerate(y):
            if torch.is_tensor(t):
                print(f"  OUT[{i}] {tuple(t.shape)}")
            else:
                print(f"  OUT[{i}] {type(t)}")
    else:
        print("  OUT", tuple(y.shape))


def collect_images(calib_dir, num):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(calib_dir, e)))
    files = sorted(files)
    if num > 0:
        files = files[:num]
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../runs/croprow_yolov8_seg/weights/best.pt")
    parser.add_argument("--calib_dir", default="../dataset/images/train")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--quant_mode", default="calib", choices=["calib", "test"])
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--out", default="quantize_result")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--keep_silu", action="store_true", help="do not replace SiLU with LeakyReLU")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    model = CroprowRawWrapper(args.model, replace_silu=not args.keep_silu)
    model.eval()
    print_output_shapes(model, args.imgsz)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    quantizer = torch_quantizer(args.quant_mode, model, (dummy,), output_dir=args.out)
    qmodel = quantizer.quant_model
    qmodel.eval()

    images = collect_images(args.calib_dir, args.num)
    print("Images:", len(images))
    if len(images) == 0:
        raise RuntimeError(f"No images found in {args.calib_dir}")

    with torch.no_grad():
        for i, p in enumerate(images):
            x = preprocess(p, args.imgsz)
            _ = qmodel(x)
            if i % 20 == 0:
                print(f"{i}/{len(images)} {p}")

    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print("Exported quant config")

    if args.quant_mode == "test" and args.deploy:
        quantizer.export_xmodel(deploy_check=False)
        print("Exported xmodel")


if __name__ == "__main__":
    main()
