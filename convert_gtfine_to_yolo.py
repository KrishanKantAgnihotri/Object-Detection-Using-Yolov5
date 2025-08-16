# convert_gtfine_to_yolo.py
"""
Script to convert Cityscapes/IDD20K gtFine JSON annotations to YOLO format labels.
Assumes gtFine structure and leftImg8bit images.
"""
import os
import json
from glob import glob
from PIL import Image

# Map your class names to YOLO class indices (must match data.yaml order)
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def convert_annotation(json_path, image_path, label_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    img = Image.open(image_path)
    w, h = img.size
    yolo_lines = []
    for obj in data.get('objects', []):
        label = obj.get('label')
        if label not in CLASS_NAME_TO_ID:
            continue
        cls_id = CLASS_NAME_TO_ID[label]
        # Each polygon is a list of points
        for polygon in obj.get('polygon', []):
            # Get bounding box from polygon
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            # Convert to YOLO format
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            bw = (x_max - x_min) / w
            bh = (y_max - y_min) / h
            yolo_lines.append(f"{cls_id} {x_center} {y_center} {bw} {bh}")
    with open(label_path, 'w') as f:
        f.write("\n".join(yolo_lines))

def process_split(split):
    gt_dir = os.path.join("idd20k_lite", "gtFine", split)
    img_dir = os.path.join("idd20k_lite", "leftImg8bit", split)
    label_dir = os.path.join("idd20k_lite", "labels", split)
    for subfolder in os.listdir(img_dir):
        img_subdir = os.path.join(img_dir, subfolder)
        gt_subdir = os.path.join(gt_dir, subfolder)
        label_subdir = os.path.join(label_dir, subfolder)
        os.makedirs(label_subdir, exist_ok=True)
        for img_file in glob(os.path.join(img_subdir, "*.png")):
            base = os.path.basename(img_file).replace("_leftImg8bit.png", "")
            json_file = os.path.join(gt_subdir, base + "_gtFine_polygons.json")
            label_file = os.path.join(label_subdir, base + ".txt")
            if os.path.exists(json_file):
                convert_annotation(json_file, img_file, label_file)

if __name__ == "__main__":
    for split in ["train", "val"]:
        process_split(split)
    print("Conversion complete. Labels are in idd20k_lite/labels/")
