import json
import os
import sys
from PIL import Image

# Add the root directory to sys.path to allow importing from src as a module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, root_dir)

from src.config.settings import DATA_PATH

# Construct the complete paths
train_annotations_path = os.path.join(DATA_PATH, 'bdd100k_labels_release', 'bdd100k', 'labels', 'bdd100k_labels_images_train.json')
val_annotations_path = os.path.join(DATA_PATH, 'bdd100k_labels_release', 'bdd100k', 'labels', 'bdd100k_labels_images_val.json')
train_images_dir = os.path.join(DATA_PATH, 'bdd100k_images_100k', 'bdd100k', 'images', '100k', 'train')
val_images_dir = os.path.join(DATA_PATH, 'bdd100k_images_100k', 'bdd100k', 'images', '100k', 'val')


# YOLOv5 data directory to save converted labels
output_train_labels_dir = './src/pages/model/yolov5/data/labels/train'
output_val_labels_dir = './src/pages/model/yolov5/data/labels/val'

# Make sure output directories exist
os.makedirs(output_train_labels_dir, exist_ok=True)
os.makedirs(output_val_labels_dir, exist_ok=True)

# Define BDD classes and map them to YOLO class IDs
class_map = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9
}

# Convert bounding box to YOLO format
def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / img_width
    y_center = ((y1 + y2) / 2.0) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

# Function to process annotations
def process_annotations(annotations_path, images_dir, output_labels_dir):
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)

    # Process each image's annotations
    for ann in annotations:
        image_name = ann['name']
        image_path = os.path.join(images_dir, image_name)
        
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping.")
            continue

        # Get image dimensions
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        # YOLO format annotation file
        label_file_path = os.path.join(output_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")

        # Open label file to write annotations
        with open(label_file_path, 'w') as label_file:
            for label in ann['labels']:
                category = label['category']
                if category not in class_map:
                    continue  # Skip if category is not in the selected classes

                class_id = class_map[category]
                bbox = label['box2d']
                x_center, y_center, width, height = convert_bbox_to_yolo_format(
                    [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], img_width, img_height
                )
                # Write to the label file in YOLO format
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Processed {image_name}")

# Process train and validation annotations
process_annotations(train_annotations_path, train_images_dir, output_train_labels_dir)
process_annotations(val_annotations_path, val_images_dir, output_val_labels_dir)
