# YOLOv8 Crop-Row Segmentation on Xilinx ZCU104 DPU

This repository documents the deployment of a **custom YOLOv8 instance-segmentation model for crop-row detection/segmentation on the Xilinx ZCU104** using **Vitis-AI 3.0** and the **DPUCZDX8G** accelerator.

Unlike the ARM-only NCNN version of this project, this implementation moves the main CNN workload to the FPGA DPU while keeping preprocessing and YOLOv8 segmentation post-processing on the ARM processor.

> **Important:** this is not a fully-DPU YOLOv8-Seg pipeline. The DPU executes the quantized CNN graph, while the ARM still performs DFL decode, confidence filtering, NMS, prototype-mask reconstruction, resizing and drawing.

---

## 1. Project objective

The objective is to evaluate whether the ZCU104 DPU can accelerate a custom YOLOv8 crop-row segmentation model while preserving acceptable segmentation quality.

The project investigates:

- training a single-class YOLOv8 segmentation model for crop rows;
- validating the original PyTorch/ONNX model on Ubuntu;
- modifying the YOLOv8 graph to make it compatible with Vitis-AI/XIR;
- INT8 post-training quantization using `vai_q_pytorch`;
- exporting and compiling an `.xmodel` for the ZCU104 DPU;
- running image and video inference on the ZCU104;
- comparing DPU inference time with ARM post-processing time;
- identifying accuracy and performance limitations introduced by graph modification and quantization.

---

## 2. Deployment architecture

The final embedded pipeline is:

```text
Input image / video frame
        |
        v
ARM preprocessing
(letterbox, RGB conversion, normalization, INT8 scaling)
        |
        v
DPUCZDX8G
(quantized YOLOv8 backbone + neck + raw segmentation head)
        |
        v
10 raw output tensors
        |
        v
ARM post-processing
DFL decode
confidence filtering
NMS
mask coefficient x prototype
mask resize / morphology
drawing
        |
        v
Result image / video
```

The DPU is responsible for the computationally expensive convolutional network, while the ARM handles operators that are not practical or supported directly by the DPU flow.

---

## 3. Hardware platform

### Xilinx ZCU104

- **Board:** Xilinx ZCU104 RevC
- **Device:** Zynq UltraScale+ MPSoC ZU7EV
- **Target DPU:** DPUCZDX8G
- **Compiler architecture:** `DPUCZDX8G_ISA1_B4096`
- **DRAM:** 2 GB
- **Target OS:** Xilinx/PetaLinux-based Vitis-AI image
- **Linux kernel:** 5.15.x Xilinx 2022.2
- **OpenCV:** 4.5.2
- **Python:** 3.9.x on the board

The tested SD-card image uses the Vitis-AI 3.0 ZCU104 runtime environment.

---

## 4. Host development environment

The host development environment used for quantization and compilation is based on:

```text
Ubuntu 20.04
Docker
Vitis-AI 3.0
Python 3.7.12
PyTorch 1.12.1
vai_q_pytorch 3.0
CPU-only quantization
```

Typical Docker environment:

```text
xilinx/vitis-ai-pytorch-cpu
```

Activate the Vitis-AI PyTorch environment:

```bash
conda activate vitis-ai-pytorch
```

Check:

```bash
python3 - <<'PY'
import torch
import pytorch_nndct

print("PyTorch:", torch.__version__)
print("pytorch_nndct: OK")
PY
```

---

## 5. Model and dataset

The original model is a custom YOLOv8 segmentation model trained for one class:

```text
crop_row
```

The model is evaluated at:

```text
input size: 640 x 640
batch size: 1
```

Example dataset layout:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Example `data.yaml`:

```yaml
path: /workspace/project/croprow_zcu104_rebuild/dataset

train: images/train
val: images/val
test: images/test

nc: 1

names:
  0: crop_row
```

---

## 6. Why the original YOLOv8-Seg graph cannot be exported directly

A standard YOLOv8 segmentation graph contains several operations that are problematic for the Vitis-AI 3.0 / XIR flow used in this project.

The initial export failed with errors such as:

```text
XIR don't support multi-outputs op ... nndct_chunk
XIR don't support multi-outputs op ... aten::unbind
XIR don't support multi-outputs op ... aten::meshgrid
XIR don't support multi-outputs op ... aten::split_with_sizes
```

Therefore, the model graph was modified before quantization.

---

## 7. DPU-friendly graph modifications

Three important modifications are applied before Vitis-AI quantization.

### 7.1 C2f: `chunk` -> slicing

YOLOv8 `C2f.forward()` normally uses tensor chunking.

This is replaced by slicing to avoid unsupported multi-output `nndct_chunk` operations.

```text
[PATCH] C2f.forward: chunk -> slicing
```

### 7.2 Segment head -> raw outputs

The standard YOLOv8 segmentation head performs decoding logic internally.

For DPU deployment, the head is changed to return only raw convolutional outputs. Grid generation, DFL decoding and final mask reconstruction are moved to ARM post-processing.

```text
[PATCH] Segment.forward -> raw outputs
```

### 7.3 SiLU -> LeakyReLU

SiLU layers are replaced with LeakyReLU for better compatibility with the selected DPU toolchain.

```text
[PATCH] SiLU -> LeakyReLU
```

In the tested crop-row model, 66 SiLU modules were replaced.

> This modification is one of the main limitations of the current implementation because changing the activation after training can change the model response and segmentation accuracy.

---

## 8. Raw output structure

After patching, the segmentation model returns **10 raw tensors**.

PyTorch/NCHW view:

```text
OUT[0]  proto  : [1, 32, 160, 160]

OUT[1]  box8   : [1, 64, 80, 80]
OUT[2]  cls8   : [1,  1, 80, 80]
OUT[3]  mask8  : [1, 32, 80, 80]

OUT[4]  box16  : [1, 64, 40, 40]
OUT[5]  cls16  : [1,  1, 40, 40]
OUT[6]  mask16 : [1, 32, 40, 40]

OUT[7]  box32  : [1, 64, 20, 20]
OUT[8]  cls32  : [1,  1, 20, 20]
OUT[9]  mask32 : [1, 32, 20, 20]
```

The three detection scales correspond to:

```text
stride 8
stride 16
stride 32
```

For YOLOv8 with `reg_max = 16`:

```text
box channels = 4 x 16 = 64
```

The 32-channel mask coefficients are later combined with the prototype tensor on the ARM.

---

## 9. PyTorch INT8 quantization

The project uses the Vitis-AI PyTorch quantizer.

### Calibration

Example:

```bash
cd quant

python3 quant_croprow.py \
    --quant_mode calib \
    --num -1
```

The tested calibration run used:

```text
908 training images
```

Successful calibration ends with:

```text
=>Exporting quant config.(quantize_result/quant_info.json)
Exported quant config
```

### Quantized test

```bash
python3 quant_croprow.py \
    --quant_mode test \
    --num 100
```

### Export deployable XModel

```bash
python3 quant_croprow.py \
    --quant_mode test \
    --num 100 \
    --deploy
```

Successful export:

```text
=>Successfully convert 'CroprowRawWrapper' to xmodel.
```

Example output:

```text
quantize_result/CroprowRawWrapper_int.xmodel
```

---

## 10. Compile for ZCU104

Compile the quantized XModel using the ZCU104 DPU architecture:

```bash
vai_c_xir \
    -x quantize_result/CroprowRawWrapper_int.xmodel \
    -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
    -o compile \
    -n croprow_yolov8
```

Successful compilation:

```text
Graph name: CroprowRawWrapper, with op num: 575
Total device subgraph number 12, DPU subgraph number 1
Compile done.
```

Output:

```text
compile/croprow_yolov8.xmodel
```

---

## 11. Copy the model to the ZCU104

Example:

```bash
scp compile/croprow_yolov8.xmodel \
    root@<ZCU104_IP>:/home/root/board_compare/compile/
```

Copy an image and inference script:

```bash
scp test.jpg infer.py \
    root@<ZCU104_IP>:/home/root/board_compare/compile/
```

Connect:

```bash
ssh root@<ZCU104_IP>
```

---

## 12. Verify the DPU

On the ZCU104:

```bash
xdputil query
```

The board should report a DPUCZDX8G instance.

The XModel can also be inspected with:

```bash
xdputil xmodel croprow_yolov8.xmodel -l
```

---

## 13. DPU tensor layout on ZCU104

The compiled model uses NHWC tensors on the board.

Observed input:

```text
[1, 640, 640, 3]
fix_point = +6
```

Observed outputs:

```text
proto  [1,160,160,32] fp= 0

box8   [1,80,80,64]   fp=+1
box16  [1,40,40,64]   fp= 0
box32  [1,20,20,64]   fp=-1

cls8   [1,80,80,1]    fp=-1
cls16  [1,40,40,1]    fp=-1
cls32  [1,20,20,1]    fp=-2

mask8  [1,80,80,32]   fp=+3
mask16 [1,40,40,32]   fp=+3
mask32 [1,20,20,32]   fp=+2
```

The inference code manually dequantizes INT8 tensors using their Vitis-AI `fix_point`.

---

## 14. Image inference on the ZCU104

Example:

```bash
python3 infer.py \
    test.jpg \
    result.jpg \
    --xmodel croprow_yolov8.xmodel
```

The ARM performs:

```text
1. Letterbox input to 640 x 640
2. Convert BGR -> RGB
3. Normalize and quantize input
4. Execute DPU runner
5. Dequantize raw outputs
6. DFL box decoding
7. Confidence filtering
8. Box NMS
9. Mask coefficient x prototype
10. Mask resize / crop
11. Optional morphology / component filtering
12. Draw mask / crop-row result
13. Save output image
```

No `imshow()` is required, which allows inference over SSH/headless operation.

---

## 15. Image performance

One of the first working runs produced:

```text
DPU:              43.3 ms
ARM postprocess: 1474.5 ms
Total:           1517.8 ms
FPS:                0.66
```

The DPU itself is therefore approximately:

```text
~23 FPS equivalent CNN inference
```

but total application speed is limited by ARM segmentation post-processing.

After reducing the number of masks/candidates:

```text
DPU:              43.2 ms
ARM postprocess:  303.3 ms
Total:            346.5 ms
FPS:                2.89
```

This significantly improved speed, but aggressive filtering also removed some valid crop rows.

---

## 16. Post-processing trade-off

Two competing goals were observed:

```text
High recall
-> retain more candidate masks
-> better coverage of crop rows
-> more overlapping masks
-> slower ARM post-processing
```

versus:

```text
Strict filtering
-> fewer masks
-> faster execution
-> cleaner visualization
-> possible loss of valid crop rows
```

Therefore, NMS and mask thresholds must be selected according to the target application rather than only maximizing FPS.

---

## 17. Video inference

The intended video flow is:

```text
ARM reads video
      |
      v
Extract frame
      |
      v
DPU inference
      |
      v
ARM segmentation post-processing
      |
      v
Draw crop rows
      |
      v
ARM writes output video
```

No GUI display is required.

For the tested ZCU104 image, MJPEG/AVI input was more reliable than some H.264 MP4 streams because the runtime image may not contain every GStreamer codec plugin.

A compatible input can be produced on Ubuntu with:

```bash
ffmpeg -i input.mp4 \
    -c:v mjpeg \
    -q:v 5 \
    -an \
    video_test_mjpg.avi
```

---

## 18. Full-HD video limitation

For a `1920 x 1080` video, the DPU time remained almost unchanged:

```text
DPU ~43.5 ms/frame
```

but ARM mask reconstruction became very expensive.

Observed experimental ranges:

```text
ARM post-processing: ~4-6 s/frame
Processing speed:    ~0.16-0.24 FPS
```

The bottleneck was mainly caused by:

- large output resolution;
- repeated mask resizing;
- multiple candidate instance masks;
- morphology and connected-component processing;
- Python/OpenCV overhead;
- video decoding/encoding.

This shows that accelerating only the CNN does not guarantee an end-to-end real-time segmentation pipeline.

---

## 19. Quantization behavior observed

During DPU testing, some class outputs became highly saturated.

Example:

```text
Candidates after confidence: 417

Score:
min  = 0.5000
max  = 1.0000
mean = 0.9551

> 0.99 = 351
```

This makes confidence thresholding less effective and increases the amount of ARM post-processing.

This behavior is one of the main reasons that DPU segmentation quality differs from the original PyTorch/NCNN implementation.

---

## 20. Comparison with ARM-only NCNN implementation

A separate ARM-only implementation of this crop-row project was also tested using NCNN.

Typical observation:

```text
ARM + NCNN YOLOv8-Seg
~2.6 s/frame
```

The ARM/NCNN implementation was slower in raw neural-network inference, but its segmentation masks were closer to the original model.

The DPU implementation reduced CNN execution to about:

```text
43 ms/frame
```

but required a modified/quantized graph and substantial ARM post-processing.

This comparison highlights an important deployment trade-off:

| Item | ARM + NCNN | DPU + VART |
|---|---|---|
| CNN acceleration | ARM only | DPU |
| Model modification | Low | High |
| INT8 quantization | Not required | Required |
| DPU inference time | N/A | ~43 ms |
| Mask post-processing | ARM | ARM |
| Segmentation fidelity | Better in tested baseline | More sensitive to quantization/patches |
| End-to-end optimization | Limited by ARM inference | Limited by ARM post-processing |

---

## 21. Why the DPU result may differ from PyTorch

The difference is not caused by one single factor.

The deployed graph differs from the original model because of:

```text
FP32 -> INT8 quantization
C2f chunk -> slicing
Segment head -> raw outputs
SiLU -> LeakyReLU
```

The most significant concern is replacing the activation function **after training**.

Therefore, a successful XModel compilation does not guarantee that the deployed model will have exactly the same segmentation behavior as the original `.pt` model.

---

## 22. Recommended validation methodology

Use the same input image through all stages:

```text
PyTorch FP32
     |
     v
ONNX FP32
     |
     v
Quantized model
     |
     v
Compiled XModel
     |
     v
ZCU104 DPU output
```

Compare:

- bounding boxes;
- class confidence;
- mask area;
- mask IoU;
- crop-row coverage;
- DPU execution time;
- ARM post-processing time;
- total execution time.

This makes it possible to identify whether degradation occurs during:

```text
export
quantization
compilation
or post-processing
```

---

## 23. Debugging checklist

### XModel cannot find a DPU subgraph

Verify the compiler log contains:

```text
DPU subgraph number 1
```

and inspect:

```bash
xdputil xmodel croprow_yolov8.xmodel -l
```

### XIR reads the image path as the XModel

Check command-line argument parsing.

Use explicit options such as:

```bash
python3 infer.py test.jpg result.jpg \
    --xmodel croprow_yolov8.xmodel
```

### `inspect.py` breaks NumPy/OpenCV

Do not name a local Python file:

```text
inspect.py
```

because it shadows the Python standard-library `inspect` module.

Rename it, for example:

```text
inspect_xmodel.py
```

### No mask appears

Check:

```text
tensor mapping
fix_point
manual dequantization
prototype layout
mask coefficients
mask threshold
letterbox coordinates
```

### Too many masks

Tune:

```text
confidence threshold
box NMS IoU
maximum candidates
mask threshold
mask IoU
component area filters
```

---

## 24. Current project status

- [x] Custom crop-row YOLOv8 segmentation model
- [x] PyTorch baseline
- [x] ONNX FP32 validation
- [x] DPU-friendly graph patching
- [x] Vitis-AI PyTorch INT8 calibration
- [x] INT8 XModel export
- [x] ZCU104 compilation
- [x] DPU subgraph execution
- [x] Image inference on ZCU104
- [x] ARM segmentation post-processing
- [x] Headless image output
- [x] Video-frame DPU inference experiments
- [x] ARM-vs-DPU performance analysis
- [ ] Fully optimized C++ post-processing
- [ ] Quantization-aware training
- [ ] ONNX quantization/compiler evaluation
- [ ] Real-time segmentation

---

## 25. Future improvements

The most promising future directions are:

### 25.1 Train a DPU-friendly model

Instead of replacing SiLU after training, train/fine-tune a model using DPU-compatible activation from the beginning.

### 25.2 Quantization-aware training

QAT may reduce the accuracy loss caused by INT8 PTQ.

### 25.3 ONNX quantization

A compatible ONNX quantization/compiler flow could preserve the original graph more closely than the patched PyTorch path.

### 25.4 C++ post-processing

Move:

```text
DFL
NMS
mask reconstruction
resize
morphology
drawing
```

from Python to C++/OpenCV to reduce ARM overhead.

### 25.5 Reduce output resolution

For video, `1280 x 720` is significantly cheaper to post-process than `1920 x 1080`.

### 25.6 Use semantic segmentation

Crop-row recognition does not always require instance-level masks.

A lightweight semantic-segmentation model could output:

```text
crop_row / background
```

directly and avoid:

```text
DFL
NMS
mask coefficients
prototype-mask reconstruction
```

This may be a better architecture for a future real-time ZCU104 implementation.

---

## 26. Key conclusion

The project successfully demonstrates that a custom YOLOv8 segmentation CNN can be quantized, compiled and executed on the **ZCU104 DPUCZDX8G**.

The FPGA DPU reduces the neural-network execution time to approximately:

```text
43 ms/frame
```

However, YOLOv8 instance segmentation still requires substantial ARM-side work. In the current implementation, end-to-end performance and segmentation quality are limited mainly by:

```text
INT8/model graph modifications
+
ARM mask reconstruction/post-processing
```

Therefore, the main lesson of this project is:

> **DPU acceleration strongly improves the CNN portion of YOLOv8-Seg, but an efficient embedded segmentation system also requires a DPU-friendly model and optimized ARM post-processing.**

---

## 27. Related repositories

ARM-only crop-row deployment:

```text
https://github.com/dgvu/yolov8_segmention_croprow
```

DPU crop-row deployment:

```text
https://github.com/dgvu/yolov8_segmention_croprow_dpu
```

These repositories provide two deployment baselines for comparing:

```text
YOLOv8-Seg + NCNN + ARM
```

and:

```text
YOLOv8-Seg + Vitis-AI + DPU + ARM post-processing
```

---

## 28. Author

Repository:

```text
https://github.com/dgvu/yolov8_segmention_croprow_dpu
```

This repository is maintained as a practical record of deploying and evaluating **YOLOv8 crop-row instance segmentation on the Xilinx ZCU104 DPU**.
