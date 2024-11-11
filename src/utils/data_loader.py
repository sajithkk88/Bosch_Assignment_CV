# src/utils/data_loader.py

import os
import json
import pandas as pd
from PIL import Image
from src.config.settings import METRICS_PATH

def load_json_data(path):
    """
    Load JSON data from a specified path.

    Parameters:
        path (str): Path to the JSON file.

    Returns:
        list or dict: Parsed JSON data.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def load_metrics():
    """
    Load precomputed metrics from metrics.json.

    Returns:
        dict: Metrics data loaded from metrics.json.
    """
    return load_json_data(METRICS_PATH)

def load_csv_data(path):
    """
    Load CSV data from a specified path.

    Parameters:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(path)

def load_image_filenames(path):
    """
    Load image filenames from a given directory.

    Parameters:
        path (str): Directory path where images are stored.

    Returns:
        list: List of image filenames with extensions like .jpg or .png.
    """
    return [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]

def get_file_size_in_mb(file_path):
    """
    Calculate the size of a file in megabytes.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        float: File size in MB, or None if the file doesn't exist.
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return None

def get_num_images(folder_path):
    """
    Count the number of images in a given directory.

    Parameters:
        folder_path (str): Directory path where images are stored.

    Returns:
        int: Number of images in the directory.
    """
    return len(load_image_filenames(folder_path))

def get_image_sample_details(folder_path):
    """
    Get resolution, bit depth, and format of a sample image from the directory.

    Parameters:
        folder_path (str): Directory path where images are stored.

    Returns:
        dict: Dictionary with resolution, bit depth, and format of a sample image,
              or default values if no images are found.
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
