from ultralytics import YOLO
import os

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    # Train the model
    # We point to 'data.yaml' which is in your main folder
    model.train(
        data='data.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16,
        device='cpu' # Change to device='cpu' if you don't have an NVIDIA GPU
    )