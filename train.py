# train.py
"""
Handles YOLOv5 training logic.
"""


import subprocess
import sys
import os

def train_yolov5(data_yaml_path):
    """
    Trains YOLOv5 using the provided data config YAML file by calling the local yolov5 repo's train.py.
    """
    python_exe = sys.executable
    yolov5_train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5', 'train.py')
    command = [python_exe, yolov5_train_py,
               '--img', '640',
               '--epochs', '10',
               '--batch', '8',
               '--data', data_yaml_path,
               '--weights', 'yolov5s.pt',
               '--project', 'runs/train',
               '--name', 'idd20k_yolov5',
               '--exist-ok']
    print('Running YOLOv5 training command:', ' '.join(command))
    subprocess.run(command, check=True)
