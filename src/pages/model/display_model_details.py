# src/pages/model/display_model_details.py

import streamlit as st
import pandas as pd
import os

def display_model_details():
    """
    Display the Model Details page with multiple dropdown options.
    Provides options to view model overview, training details, and training results.
    """

    # Dropdown menu for model details options
    model_option = st.selectbox(
        "Select Details",
        ["Model Overview", "Training Details", "Training and Validation Results", "Inferencing on Test Data"]
    )

    # Display the appropriate details based on user selection
    if model_option == "Model Overview":
        st.markdown("### Model Overview")
        training_details = {
            "Release Date": "June 9, 2020",
            "Framework": "PyTorch",
            "Developer": "Ultralytics",
            "License": "GNU General Public License v3.0 (GPL-3.0)",
            "Model Variants": "YOLOv5s, YOLOv5m, YOLOv5l, YOLOv5x",
            "Architecture": "CSP-Darknet backbone with PANet neck and YOLO head",
             "Performance": "Real-time inference; High mAP@0.5 and mAP@0.5:0.95",
            "Pretrained Models": "COCO dataset (80 classes)",
            "Supported Inputs": "Images, videos, directories of images",
            "Export Formats": "ONNX, TorchScript, CoreML, TensorRT, OpenVINO",
            "Ease of Use": "Simplified setup, training, validation, and inference scripts",
            "Supported Hardware": "CPU, GPU (NVIDIA CUDA), TPU",
            "Use Cases": "Object detection, tracking, segmentation, and instance localization",
            "GitHub Link":"https://github.com/ultralytics/yolov5"
        }
        training_details_df = pd.DataFrame(list(training_details.items()), columns=["Parameter", "Value"])
        st.table(training_details_df)      
        
        # Directory where result images are stored
        result_dir = os.path.join("src", "pages", "model", "yolov5")
      
        # List of result images to display
        images = ["yolov5_performance.PNG","YOLOv5-architecture.PNG"]

        # Display each image if it exists in the specified directory
        for image in images:
            image_path = os.path.join(result_dir, image)
            if os.path.exists(image_path):
                st.image(image_path, caption=image)
            else:
                st.warning(f"{image} not found.")

    elif model_option == "Training Details":
        st.markdown("### Training Details")
        # Data for the table
        data = {
        "Parameter": [
        "Model", "Epochs", "Batch Size", "Image Size", "Learning Rate (lr0)", "Learning Rate Final (lrf)", 
        "Momentum", "Weight Decay", "Warmup Epochs", "Warmup Momentum", "Warmup Bias LR", "Box Loss Gain",
        "Class Loss Gain", "Class Positive Weight (cls_pw)", "Object Loss Gain", "Object Positive Weight (obj_pw)",
        "IoU Threshold (iou_t)", "Anchor Threshold (anchor_t)", "Focal Loss Gamma (fl_gamma)", "HSV Hue (hsv_h)",
        "HSV Saturation (hsv_s)", "HSV Value (hsv_v)", "Degrees", "Translate", "Scale", "Shear", "Perspective",
        "Flip Up-Down (flipud)", "Flip Left-Right (fliplr)", "Mosaic", "Mixup", "Copy-Paste", "Rectangular Training",
        "Save Checkpoints", "No Validation", "No Auto-Anchor", "No Plots", "Device", "Optimizer", "Sync Batch Norm",
        "Workers", "Project", "Name", "Optimizer Settings", "Training Command"
        ],
        "Value": [
        "YOLOv5 Small", 50, 16, 640, 0.01, 0.01, 0.937, 0.0005, 3.0, 0.8, 0.1, 0.05, 0.5, 1.0, 1.0, 1.0,
        0.2, 4.0, 0.0, 0.015, 0.7, 0.4, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, "False", "True",
        "False", "False", "False", "GPU (6GB Nvidia GeForce RTX 3050)", "SGD", "False", 8, "runs/train", 
        "bdd100k_finetune_epoch50_batch16", "lr0, lrf, momentum, weight_decay", 
        "python train.py --weights yolov5s.pt --data data/bdd100k.yaml --img 640 --epochs 50 --batch-size 16 --name bdd100k_finetune --device 0"
        ],
        "Description": [
        "YOLOv5 model architecture being used.", "Number of training epochs.", "Number of images per batch.", 
        "Input image dimensions (640x640).", "Initial learning rate for training.", "Final learning rate after the scheduler adjusts.", 
        "Momentum factor for the SGD optimizer.", "Regularization term for reducing model complexity and overfitting.", 
        "Number of epochs for the learning rate warmup.", "Initial momentum for warmup period.", "Initial learning rate for bias during warmup.",
        "Loss gain factor for bounding box regression.", "Loss gain factor for classification.", "Class positive weight.", 
        "Loss gain factor for objectness score.", "Object positive weight.", "IoU threshold for objectness score.", 
        "Anchor-matching threshold.", "Focal loss gamma for adjusting the balance of easy/hard examples.", 
        "HSV hue augmentation value.", "HSV saturation augmentation.", "HSV brightness augmentation.", 
        "Degree of rotation augmentation.", "Translation factor for data augmentation.", "Scaling factor for data augmentation.",
        "Shearing factor for data augmentation.", "Perspective transformation for data augmentation.", 
        "Probability of vertical flipping during augmentation.", "Probability of horizontal flipping during augmentation.",
        "Mosaic augmentation probability.", "Mixup augmentation probability.", "Copy-paste augmentation probability.",
        "Whether to use rectangular training shapes.", "Save checkpoints during training.", "Disable validation during training.",
        "Disable automatic anchor resizing.", "Disable plot generation.", "Device for training, 0 for GPU (6GB Nvidia GeForce RTX 3050).", 
        "Optimizer for training, here using SGD.", "Synchronize batch normalization across devices.", "Number of data loading workers.",
        "Directory for saving training runs.", "Experiment name for saving checkpoints and logs.", 
        "Various hyperparameters for optimization.", 
        "Command for training the model."
        ]
        }

        # Create DataFrame
        table_df = pd.DataFrame(data)

        # Display table in Streamlit
        st.markdown("### Parameter Summary Table")
        st.table(table_df)

        
        # Directory where result images are stored
        result_dir = os.path.join("src", "pages", "model", "yolov5")

        # List of result images to display
        images = ["training_snapshot.png"]

        # Display each image if it exists in the specified directory
        for image in images:
            image_path = os.path.join(result_dir, image)
            if os.path.exists(image_path):
                st.image(image_path, caption=image)
            else:
                st.warning(f"{image} not found.")
        

    elif model_option == "Training and Validation Results":
        st.markdown("### Training Results")

        # Directory where result images are stored
        result_dir = os.path.join("src", "pages", "model", "training_bdd100k_finetune_epoch50_batch16")

        # List all files in the directory
        all_files = os.listdir(result_dir)
        
        # Supported image extensions
        supported_extensions = [".png", ".jpg", ".jpeg"]

        # Filter only supported image files
        images = [file for file in all_files if os.path.splitext(file)[1].lower() in supported_extensions]

        # Display each image if it exists in the specified directory
        for image in images:
            image_path = os.path.join(result_dir, image)
            if os.path.exists(image_path):
                st.image(image_path, caption=image)
            else:
                st.warning(f"{image} not found.")
                
    elif model_option == "Inferencing on Test Data":
        st.markdown("### Inference on Test Data Results")

        # Directory where result images are stored
        result_dir = os.path.join("src", "pages", "model", "test_results")
        
        # List all files in the directory
        all_files = os.listdir(result_dir)
        
        # Supported image extensions
        supported_extensions = [".png", ".jpg", ".jpeg"]

        # Filter only supported image files
        images = [file for file in all_files if os.path.splitext(file)[1].lower() in supported_extensions]

        # Display each image if it exists in the specified directory
        for image in images:
            image_path = os.path.join(result_dir, image)
            if os.path.exists(image_path):
                st.image(image_path, caption=image)
            else:
                st.warning(f"{image} not found.")

if __name__ == "__main__":
    display_model_details()
