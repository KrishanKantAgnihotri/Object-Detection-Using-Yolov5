# Entry point for the object detection pipeline
# Usage: python main.py

from data_pipeline import prepare_data
from train import train_yolov5

if __name__ == "__main__":
    # Step 1: Prepare data
    data_config = prepare_data()
    # Step 2: Train YOLOv5
    train_yolov5(data_config)
