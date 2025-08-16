# data_pipeline.py
import os
import yaml

"""
Handles dataset downloading, loading, and preprocessing for YOLOv5.
"""

def prepare_data():
    """
    Prepares the IDD20K dataset for YOLOv5 training.
    Generates a YOLOv5 data.yaml file and returns its path.
    """
    dataset_dir = os.path.abspath("idd20k_lite")
    leftImg8bit = os.path.join(dataset_dir, "leftImg8bit")
    gtFine = os.path.join(dataset_dir, "gtFine")

    # Paths for train/val images
    train_images = os.path.join(leftImg8bit, "train")
    val_images = os.path.join(leftImg8bit, "val")
    train_labels = os.path.join(gtFine, "train")
    val_labels = os.path.join(gtFine, "val")

    # Class names for IDD20K (example, update as per actual dataset)
    # For demo, using a few common classes. Replace with actual classes if available.
    names = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

    # Generate data.yaml for YOLOv5
    data_yaml = {
        'train': train_images.replace('\\', '/'),
        'val': val_images.replace('\\', '/'),
        'nc': len(names),
        'names': names
    }
    data_yaml_path = os.path.abspath("data.yaml")
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    print(f"Generated YOLOv5 data.yaml at {data_yaml_path}")
    return data_yaml_path
