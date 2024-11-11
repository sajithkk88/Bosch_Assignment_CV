# src/pages/data_analysis/dataset_overview.py

import streamlit as st
import pandas as pd
from src.utils.data_loader import load_metrics

def display_dataset_overview():
    st.subheader("Dataset Overview")
    
    # Load metrics from metrics.json
    metrics = load_metrics()

    # Dropdown for train, val, or test selection
    data_option = st.selectbox("Select Dataset Subset", ["Train", "Validation", "Test"])

    # Display dataset information based on user selection
    if data_option == "Train":
        # Bold text with a specific color using HTML
        st.markdown("<h4 style='color: teal; font-weight: bold;'>Overview of Train Dataset</h4>", unsafe_allow_html=True)
        dataset_info = metrics["dataset_overview"]["train"]

        # Display Basic Information Table
        st.markdown("<h5 style='color: black; font-weight: bold;'>Basic Information</h5>", unsafe_allow_html=True)
        basic_info = {
            "Number of Images": dataset_info["num_images"],
            "JSON File Size (MB)": round(dataset_info.get("json_file_size_MB", 0), 2)
        }
        basic_info_df = pd.DataFrame(list(basic_info.items()), columns=["Metric", "Value"])
        st.table(basic_info_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
        ]))

        # Display Sample Image Details Table
        st.markdown("<h5 style='color: black; font-weight: bold;'>Sample Image Details</h5>", unsafe_allow_html=True)
        image_sample = dataset_info["image_sample"]
        image_details = {
            "Resolution": f"{image_sample['resolution'][0]} x {image_sample['resolution'][1]}",
            "Bit Depth": image_sample["bit_depth"],
            "Format": image_sample["format"]
        }
        image_details_df = pd.DataFrame(list(image_details.items()), columns=["Attribute", "Value"])
        st.table(image_details_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
        ]))

    elif data_option == "Validation":
        st.markdown("<h4 style='color: teal; font-weight: bold;'>Overview of Validation Dataset</h4>", unsafe_allow_html=True)
        dataset_info = metrics["dataset_overview"]["val"]

        # Display Basic Information Table
        st.markdown("<h5 style='color: black; font-weight: bold;'>Basic Information</h5>", unsafe_allow_html=True)
        basic_info = {
            "Number of Images": dataset_info["num_images"],
            "JSON File Size (MB)": round(dataset_info.get("json_file_size_MB", 0), 2)
        }
        basic_info_df = pd.DataFrame(list(basic_info.items()), columns=["Metric", "Value"])
        st.table(basic_info_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
        ]))

        # Display Sample Image Details Table
        st.markdown("<h5 style='color: black; font-weight: bold;'>Sample Image Details</h5>", unsafe_allow_html=True)
        image_sample = dataset_info["image_sample"]
        image_details = {
            "Resolution": f"{image_sample['resolution'][0]} x {image_sample['resolution'][1]}",
            "Bit Depth": image_sample["bit_depth"],
            "Format": image_sample["format"]
        }
        image_details_df = pd.DataFrame(list(image_details.items()), columns=["Attribute", "Value"])
        st.table(image_details_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
        ]))

    else: 
        st.markdown("<h4 style='color: teal; font-weight: bold;'>Overview of Train Dataset</h4>", unsafe_allow_html=True)
        dataset_info = metrics["dataset_overview"]["test"]

        # Display Basic Information Table
        st.markdown("<h5 style='color: black; font-weight: bold;'>Basic Information</h5>", unsafe_allow_html=True)
        basic_info = {
            "Number of Images": dataset_info["num_images"]
        }
        basic_info_df = pd.DataFrame(list(basic_info.items()), columns=["Metric", "Value"])
        st.table(basic_info_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
        ]))

        # Display Sample Image Details Table
        st.markdown("<h5 style='color: black; font-weight: bold;'>Sample Image Details</h5>", unsafe_allow_html=True)
        image_sample = dataset_info["image_sample"]
        image_details = {
            "Resolution": f"{image_sample['resolution'][0]} x {image_sample['resolution'][1]}",
            "Bit Depth": image_sample["bit_depth"],
            "Format": image_sample["format"]
        }
        image_details_df = pd.DataFrame(list(image_details.items()), columns=["Attribute", "Value"])
        st.table(image_details_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
        ]))

if __name__ == "__main__":
    display_dataset_overview()
