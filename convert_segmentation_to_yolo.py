# convert_segmentation_to_yolo.py
"""
Convert IDD20K segmentation masks (_label.png) to YOLO object detection labels.
Each connected component of a class in the mask becomes a bounding box.
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label

# Class mapping (update as needed to match your data.yaml)
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
CLASS_IDS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Only use classes you want to detect (currently all classes)
USE_CLASSES = set(range(len(CLASS_NAMES)))


def mask_to_boxes(mask, class_id):
    """Extract bounding boxes for a given class_id from the mask"""
    boxes = []
    mask_bin = (mask == class_id).astype(np.uint8)
    labeled, n = cc_label(mask_bin)
    for i in range(1, n + 1):
        pos = np.where(labeled == i)
        if pos[0].size == 0 or pos[1].size == 0:
            continue
        y_min, y_max = pos[0].min(), pos[0].max()
        x_min, x_max = pos[1].min(), pos[1].max()
        boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def convert_mask_to_yolo(mask_path, image_path, label_path):
    """Convert segmentation mask to YOLO labels and save as .txt"""
    mask = np.array(Image.open(mask_path))
    h, w = mask.shape
    yolo_lines = []
    for class_id in USE_CLASSES:
        boxes = mask_to_boxes(mask, class_id)
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    # Only save if labels exist
    if yolo_lines:
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_lines))


def process_split(split):
    mask_root = os.path.join("idd20k_lite", "gtFine", split)
    img_root = os.path.join("idd20k_lite", "images", split)
    label_root = os.path.join("idd20k_lite", "labels", split)

    total_labels = 0

    for subfolder in os.listdir(mask_root):
        mask_dir = os.path.join(mask_root, subfolder)
        img_dir = os.path.join(img_root, subfolder)
        label_dir = os.path.join(label_root, subfolder)
        os.makedirs(label_dir, exist_ok=True)

        for mask_file in os.listdir(mask_dir):
            if not mask_file.endswith("_label.png"):
                continue
            base = mask_file.replace("_label.png", "")


            img_path = os.path.join(img_dir, base + "_image.jpg")
            label_path = os.path.join(label_dir, base + "_image.txt")

            if os.path.exists(img_path):
                convert_mask_to_yolo(
                    mask_path=os.path.join(mask_dir, mask_file),
                    image_path=img_path,
                    label_path=label_path
                )
                total_labels += 1

    print(f"Generated {total_labels} label files for {split} split")


if __name__ == "__main__":
    for split in ["train", "val"]:
        process_split(split)
    print("Segmentation masks converted to YOLO labels in idd20k_lite/labels/")
