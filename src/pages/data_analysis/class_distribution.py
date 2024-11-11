# src/pages/data_analysis/class_distribution.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_loader import load_metrics

def display_class_distribution():
    st.subheader("Class Distribution")
    
    # Load precomputed metrics from metrics.json
    metrics = load_metrics()

    # Dropdown to select dataset subset
    data_option = st.selectbox("Select Dataset Subset", ["Train", "Validation"])

    # Display dataset information based on user selection
    if data_option == "Train":
        st.markdown("<h4 style='color: teal; font-weight: bold;'>Class Distribution - Train Dataset</h4>", unsafe_allow_html=True)
        dataset_info = metrics["dataset_overview"]["train"]
        all_category_counts = metrics["all_category_counts"]["train"]
        detection_classes = metrics["class_distribution"]["train"]
    else:
        st.markdown("<h4 style='color: teal; font-weight: bold;'>Class Distribution - Validation Dataset</h4>", unsafe_allow_html=True)
        dataset_info = metrics["dataset_overview"]["val"]
        all_category_counts = metrics["all_category_counts"]["val"]
        detection_classes = metrics["class_distribution"]["val"]

    # Basic information table
    total_objects_all_classes = sum(all_category_counts.values())
    total_objects_detection_classes = sum(detection_classes.values())
    basic_info = {
        "Total Number of Images": dataset_info["num_images"],
        "Total Objects (All Classes)": total_objects_all_classes,
        "Total Objects in Detection Classes": total_objects_detection_classes,
    }
    basic_info_df = pd.DataFrame(list(basic_info.items()), columns=["Metric", "Value"])
    st.table(basic_info_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
        {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
    ]))

    # Display count of each detection class separately
    st.markdown("<h5 style='color: black; font-weight: bold;'>Count of Each Detection Class</h5>", unsafe_allow_html=True)
    detection_class_df = pd.DataFrame(detection_classes.items(), columns=["Class", "Count"])
    st.table(detection_class_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
        {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
    ]))

    # Histogram for 10 detection classes
    st.markdown("<h5 style='color: black; font-weight: bold;'>Histogram of Detection Classes</h5>", unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    plt.bar(detection_class_df["Class"], detection_class_df["Count"], color="#4c72b0")
    plt.xlabel("Class", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Display count of all classes if available
    st.markdown("<h5 style='color: black; font-weight: bold;'>Count of All Classes in Dataset</h5>", unsafe_allow_html=True)
    all_classes_df = pd.DataFrame(all_category_counts.items(), columns=["Class", "Count"])
    st.table(all_classes_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
        {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
    ]))

    # Histogram for all classes
    st.markdown("<h5 style='color: black; font-weight: bold;'>Histogram of All Classes</h5>", unsafe_allow_html=True)
    plt.figure(figsize=(12, 8))
    plt.bar(all_classes_df["Class"], all_classes_df["Count"], color="#2ca02c")
    plt.xlabel("Class", fontweight="bold")
    plt.ylabel("Count", fontweight="bold")
    plt.xticks(rotation=45)
    st.pyplot(plt)

if __name__ == "__main__":
    display_class_distribution()
