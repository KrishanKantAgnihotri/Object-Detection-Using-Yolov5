# """
# """ Handles YOLOv5 training logic. """ import subprocess import sys import os def train_yolov5(data_yaml_path): """ Trains YOLOv5 using the provided data config YAML file by calling the local yolov5 repo's train.py. """ python_exe = sys.executable yolov5_train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5', 'train.py') command = [python_exe, yolov5_train_py, '--img', '640', '--epochs', '10', '--batch', '8', '--data', data_yaml_path, '--weights', 'yolov5s.pt', '--project', 'runs/train', '--name', 'idd20k_yolov5', '--exist-ok'] print('Running YOLOv5 training command:', ' '.join(command)) subprocess.run(command)
# """
# train.py
"""
Handles YOLOv5 training logic (optimized for low-resource CPU training).
"""

import subprocess
import sys
import os

def train_yolov5(data_yaml_path):
    """
    Trains YOLOv5 using the provided data config YAML file
    Optimized for laptops with CPU and 8 GB RAM.
    """
    python_exe = sys.executable
    yolov5_train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5', 'train.py')

    command = [
        python_exe, yolov5_train_py,
        '--img', '320',               # smaller image size (faster, less RAM)
        '--epochs', '3',              # just for testing, increase later
        '--batch', '4',               # lower batch to fit in RAM
        '--data', data_yaml_path,
        '--weights', 'yolov5n.pt',    # nano model (fastest, least memory)
        '--project', 'runs/train',
        '--name', 'idd20k_yolov5',
        '--workers', '2',             # limit dataloader workers
        '--device', 'cpu',            # force CPU training
        '--exist-ok'
    ]

    print('Running YOLOv5 training command:', ' '.join(command))
    subprocess.run(command)


if __name__ == "__main__":
    # Example usage (change path if needed)
    train_yolov5("data.yaml")
