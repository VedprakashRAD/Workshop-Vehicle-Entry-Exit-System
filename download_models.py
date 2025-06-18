import os
import requests
import torch
from ultralytics import YOLO
import easyocr

def download_models():
    """Download required models for vehicle detection and license plate recognition"""
    print("Downloading and setting up models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Download YOLOv8n model
    print("Downloading YOLOv8n model...")
    try:
        # Using ultralytics to download the model
        model = YOLO('yolov8n.pt')
        print("YOLOv8n model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading YOLOv8n model: {e}")
    
    # Initialize EasyOCR to download its models
    print("Downloading EasyOCR models (this may take a while)...")
    try:
        # This will download the necessary recognition models
        reader = easyocr.Reader(['en'])
        print("EasyOCR models downloaded successfully!")
    except Exception as e:
        print(f"Error downloading EasyOCR models: {e}")
    
    print("\nModel setup completed!")
    print("\nTo run the application, use:")
    print("  python app.py")
    print("\nTo test with a video file, use:")
    print("  python test_video.py path/to/video.mp4")

if __name__ == "__main__":
    download_models() 