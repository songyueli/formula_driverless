# ML — YOLO Cone Detection Training

This directory is a standalone Python environment. It is completely separate from
the C++ CMake build — nothing here is compiled or linked by CMake.

## Workflow

```
train & fine-tune (here, Python)  →  export to ONNX  →  run in C++ via TensorRT
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Put your labeled cone images in `data/` (YOLO format: images/ + labels/).
2. Edit the dataset path in `train.py`.
3. Run training:
   ```bash
   python train.py
   ```
4. The script exports a `.onnx` file at the end. Copy it to the car's Jetson and
   load it via TensorRT in the C++ `perception` process.

## Cone classes

| Class   | Color  | Track position |
|---------|--------|----------------|
| 0       | Blue   | Left boundary  |
| 1       | Yellow | Right boundary |
| 2       | Orange | Entry / exit   |
