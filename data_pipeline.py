import os
import yaml
def prepare_data():
    dataset_dir = os.path.abspath("idd20k_lite")

    # Paths for train/val images
    train_images = os.path.join(dataset_dir, "images", "train")
    val_images = os.path.join(dataset_dir, "images", "val")

    #  Correct labels path
    train_labels = os.path.join(dataset_dir, "labels", "train")
    val_labels = os.path.join(dataset_dir, "labels", "val")

    names = [
        "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
        "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
    ]

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
