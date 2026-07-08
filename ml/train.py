# YOLO cone detection — training script
#
# Steps:
#   TODO 1: Choose a YOLO variant (e.g. YOLOv8n for speed, YOLOv8m for accuracy)
#            and decide whether to train from scratch or fine-tune a pretrained checkpoint.
#   TODO 2: Prepare the dataset in YOLO format:
#            data/
#              images/train/  *.jpg
#              images/val/    *.jpg
#              labels/train/  *.txt  (one line per cone: class cx cy w h, normalized)
#              labels/val/    *.txt
#            Write a dataset.yaml pointing to these paths with 3 classes:
#            names: [blue, yellow, orange]
#   TODO 3: Run fine-tuning:
#            from ultralytics import YOLO
#            model = YOLO("yolov8n.pt")
#            model.train(data="ml/data/dataset.yaml", epochs=100, imgsz=1440)
#   TODO 4: Validate on the val split and inspect the confusion matrix.
#   TODO 5: Export to ONNX for deployment:
#            model.export(format="onnx", imgsz=1440, simplify=True)
#            The exported file (best.onnx) is what the C++ perception process loads
#            via TensorRT on the Jetson AGX Orin.
