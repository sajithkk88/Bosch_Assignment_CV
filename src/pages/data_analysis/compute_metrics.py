# src/pages/data_analysis/compute_metrics.py

import os
import json
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
from src.config.settings import (
    TRAIN_LABELS_PATH,
    VAL_LABELS_PATH,
    TRAIN_IMAGES_PATH,
    TEST_IMAGES_PATH,
    VAL_IMAGES_PATH,
    METRICS_PATH
)
from src.utils.data_loader import load_json_data, load_image_filenames


def get_num_images(folder_path):
    """
    Get the number of images in a folder.

    Parameters:
        folder_path (str): Directory path where images are stored.

    Returns:
        int: Number of images in the directory.
    """
    return len(load_image_filenames(folder_path))


def get_image_sample_details(folder_path):
    """
    Extract resolution, bit depth, and format from a sample image.

    Parameters:
        folder_path (str): Directory path where images are stored.

    Returns:
        dict: Dictionary with resolution, bit depth, and format of a sample image.
    """
    for filename in load_image_filenames(folder_path):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as img:
            resolution = img.size
            bit_depth = img.bits if "bits" in img.info else 8  # Default to 8 if not available
            image_format = img.format
        return {
            "resolution": resolution,
            "bit_depth": bit_depth,
            "format": image_format
        }
    return {"resolution": None, "bit_depth": None, "format": None}


def get_file_size_in_mb(file_path):
    """
    Return the size of a file in MB.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        float: File size in MB.
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return None


def convert_to_serializable(obj):
    """
    Convert numpy data types and other non-serializable objects to JSON-compatible types.

    Parameters:
        obj (Any): Object to convert.

    Returns:
        Any: JSON-serializable object.
    """
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def count_all_categories(data):
    """
    Count occurrences of each category in the data.

    Parameters:
        data (list): List of data entries containing labels.

    Returns:
        Counter: Counts of each category.
    """
    class_counter = Counter()
    for item in data:
        for label in item.get("labels", []):
            category = label.get("category")
            class_counter[category] += 1
    return class_counter


def count_detection_classes(data, detection_classes):
    """
    Count occurrences of specific classes in the data.

    Parameters:
        data (list): List of data entries containing labels.
        detection_classes (dict): Dictionary of detection classes to count.

    Returns:
        Counter: Counts of specified detection classes.
    """
    class_counter = Counter()
    for item in data:
        for label in item.get("labels", []):
            category = label.get("category")
            if category in detection_classes:
                class_counter[category] += 1
    return class_counter


def bounding_box_analysis(data, detection_classes):
    """
    Analyze bounding box area and aspect ratio for each class.

    Parameters:
        data (list): List of data entries containing labels and bounding box information.
        detection_classes (dict): Dictionary of detection classes to analyze.

    Returns:
        dict: Area and aspect ratio statistics for each class.
    """
    area_stats = defaultdict(list)
    aspect_ratio_stats = defaultdict(list)

    for item in data:
        for label in item.get("labels", []):
            if "box2d" in label:
                category = label.get("category")
                if category in detection_classes:
                    box = label["box2d"]
                    width = box["x2"] - box["x1"]
                    height = box["y2"] - box["y1"]
                    area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    area_stats[category].append(area)
                    aspect_ratio_stats[category].append(aspect_ratio)

    return {
        "area_summary": {k: {"mean": np.mean(v), "std": np.std(v), "min": min(v), "max": max(v)} for k, v in area_stats.items()},
        "aspect_ratio_summary": {k: {"mean": np.mean(v), "std": np.std(v), "min": min(v), "max": max(v)} for k, v in aspect_ratio_stats.items()}
    }


def object_attributes_analysis(data, detection_classes):
    """
    Analyze object attributes like occlusion, truncation, and traffic light color.

    Parameters:
        data (list): List of data entries containing labels and attributes.
        detection_classes (dict): Dictionary of detection classes to analyze.

    Returns:
        dict: Counts of occlusion, truncation, and traffic light colors.
    """
    occlusion_counts = Counter()
    truncation_counts = Counter()
    traffic_light_colors = Counter()

    for item in data:
        for label in item.get("labels", []):
            category = label.get("category")
            if category in detection_classes:
                attributes = label.get("attributes", {})
                if attributes.get("occluded"):
                    occlusion_counts[category] += 1
                if attributes.get("truncated"):
                    truncation_counts[category] += 1
                if category == "traffic light":
                    color = attributes.get("trafficLightColor", "none")
                    traffic_light_colors[color] += 1

    return {
        "occlusion_counts": occlusion_counts,
        "truncation_counts": truncation_counts,
        "traffic_light_colors": traffic_light_colors
    }


def scene_context_analysis(data, detection_classes):
    """
    Analyze scene context, showing class distribution by scene type.

    Parameters:
        data (list): List of data entries containing scene attributes.
        detection_classes (dict): Dictionary of detection classes to analyze.

    Returns:
        dict: Scene context counts for each class.
    """
    scene_class_counts = defaultdict(Counter)
    for item in data:
        scene = item["attributes"].get("scene", "undefined")
        for label in item.get("labels", []):
            category = label.get("category")
            if category in detection_classes:
                scene_class_counts[scene][category] += 1

    return {"scene_class_counts": scene_class_counts}


def object_density_analysis(data):
    """
    Analyze object density per image.

    Parameters:
        data (list): List of data entries containing labels.

    Returns:
        dict: Average, min, max, and standard deviation of objects per image.
    """
    num_objects_per_image = [len(item.get("labels", [])) for item in data]
    return {
        "average_objects_per_image": np.mean(num_objects_per_image),
        "min_objects_per_image": np.min(num_objects_per_image),
        "max_objects_per_image": np.max(num_objects_per_image),
        "std_objects_per_image": np.std(num_objects_per_image)
    }


def compute_and_save_metrics():
    """
    Compute and save metrics for train, val, and test datasets.

    This function calculates various metrics related to the dataset and saves
    them in a JSON file specified by the METRICS_PATH.
    """
    detection_classes = {
        "pedestrian": 1, "rider": 2, "car": 3, "truck": 4, "bus": 5, 
        "train": 6, "motorcycle": 7, "bicycle": 8, "traffic light": 9, "traffic sign": 10
    }

    # Load JSON data
    train_data = load_json_data(TRAIN_LABELS_PATH)
    val_data = load_json_data(VAL_LABELS_PATH)

    # Compute metrics
    metrics = {
        "dataset_overview": {
            "train": {
                "num_images": get_num_images(TRAIN_IMAGES_PATH),
                "image_sample": get_image_sample_details(TRAIN_IMAGES_PATH),
                "json_file_size_MB": get_file_size_in_mb(TRAIN_LABELS_PATH)
            },
            "val": {
                "num_images": get_num_images(VAL_IMAGES_PATH),
                "image_sample": get_image_sample_details(VAL_IMAGES_PATH),
                "json_file_size_MB": get_file_size_in_mb(VAL_LABELS_PATH)
            },
            "test": {
                "num_images": get_num_images(TEST_IMAGES_PATH),
                "image_sample": get_image_sample_details(TEST_IMAGES_PATH)
            }
        },
        "all_category_counts": {
            "train": count_all_categories(train_data),
            "val": count_all_categories(val_data)
        },
        "class_distribution": {
            "train": count_detection_classes(train_data, detection_classes),
            "val": count_detection_classes(val_data, detection_classes)
        },
        "bounding_box_analysis": {
            "train": bounding_box_analysis(train_data, detection_classes),
            "val": bounding_box_analysis(val_data, detection_classes)
        },
        "object_attributes": {
            "train": object_attributes_analysis(train_data, detection_classes),
            "val": object_attributes_analysis(val_data, detection_classes)
        },
        "scene_context": {
            "train": scene_context_analysis(train_data, detection_classes),
            "val": scene_context_analysis(val_data, detection_classes)
        },
        "object_density": {
            "train": object_density_analysis(train_data),
            "val": object_density_analysis(val_data)
        }
    }

    # Convert metrics to serializable types
    serializable_metrics = convert_to_serializable(metrics)

    # Save metrics to JSON file
    with open(METRICS_PATH, "w") as f:
        json.dump(serializable_metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")


# Run the function
if __name__ == "__main__":
    compute_and_save_metrics()
