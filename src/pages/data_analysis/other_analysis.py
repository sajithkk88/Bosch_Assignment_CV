# src/pages/data_analysis/other_analysis.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_loader import load_metrics


def display_other_analysis():
    """
    Display other analysis options for dataset subsets, such as Bounding Box Analysis,
    Object Attributes, Scene Context, and Object Density.
    """
    st.subheader("Other Analysis")

    # Load precomputed metrics from metrics.json
    metrics = load_metrics()

    # First dropdown to select dataset subset (Train or Validation)
    data_option = st.selectbox("Select Dataset Subset", ["Train", "Validation"])
    subset_key = "train" if data_option == "Train" else "val"  # Map dropdown choice to metrics.json key

    # Second dropdown to select type of analysis
    analysis_option = st.selectbox(
        "Select Analysis Type",
        ["Bounding Box Analysis", "Object Attributes", "Scene Context", "Object Density"]
    )

    # Display based on selected analysis type
    if analysis_option == "Bounding Box Analysis":
        st.markdown(f"<h4 style='color: teal; font-weight: bold;'>{data_option} - Bounding Box Analysis</h4>", unsafe_allow_html=True)
        bounding_box_metrics = metrics["bounding_box_analysis"][subset_key]
        area_summary = bounding_box_metrics["area_summary"]
        aspect_ratio_summary = bounding_box_metrics["aspect_ratio_summary"]

        # Display bounding box area summary
        st.markdown("<h5 style='color: black; font-weight: bold;'>Bounding Box Area Summary</h5>", unsafe_allow_html=True)
        area_df = pd.DataFrame(area_summary).transpose().applymap(lambda x: round(x, 2))
        st.table(area_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
        ]))

        # Display aspect ratio summary
        st.markdown("<h5 style='color: black; font-weight: bold;'>Bounding Box Aspect Ratio Summary</h5>", unsafe_allow_html=True)
        aspect_ratio_df = pd.DataFrame(aspect_ratio_summary).transpose().applymap(lambda x: round(x, 2))
        st.table(aspect_ratio_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
        ]))

    elif analysis_option == "Object Attributes":
        st.markdown(f"<h4 style='color: teal; font-weight: bold;'>{data_option} - Object Attributes</h4>", unsafe_allow_html=True)
        object_attributes = metrics["object_attributes"][subset_key]
        occlusion_counts = object_attributes["occlusion_counts"]
        truncation_counts = object_attributes["truncation_counts"]
        traffic_light_colors = object_attributes["traffic_light_colors"]

        # Display occlusion counts
        st.markdown("<h5 style='color: black; font-weight: bold;'>Occlusion Counts</h5>", unsafe_allow_html=True)
        occlusion_df = pd.DataFrame(occlusion_counts.items(), columns=["Class", "Occlusion Count"])
        st.table(occlusion_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
        ]))

        # Display truncation counts
        st.markdown("<h5 style='color: black; font-weight: bold;'>Truncation Counts</h5>", unsafe_allow_html=True)
        truncation_df = pd.DataFrame(truncation_counts.items(), columns=["Class", "Truncation Count"])
        st.table(truncation_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
        ]))

        # Display traffic light color distribution
        st.markdown("<h5 style='color: black; font-weight: bold;'>Traffic Light Colors</h5>", unsafe_allow_html=True)
        traffic_light_colors_df = pd.DataFrame(traffic_light_colors.items(), columns=["Color", "Count"])
        st.table(traffic_light_colors_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#e8e8e8")]}
        ]))

    elif analysis_option == "Scene Context":
        st.markdown(f"<h4 style='color: teal; font-weight: bold;'>{data_option} - Scene Context Analysis</h4>", unsafe_allow_html=True)
        scene_context = metrics["scene_context"][subset_key]["scene_class_counts"]
        scene_df = pd.DataFrame(scene_context).transpose().fillna(0).astype(int)
        st.table(scene_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f0f0f0")]}
        ]))

    elif analysis_option == "Object Density":
        st.markdown(f"<h4 style='color: teal; font-weight: bold;'>{data_option} - Object Density</h4>", unsafe_allow_html=True)
        density_data = metrics["object_density"][subset_key]
        density_df = pd.DataFrame(list(density_data.items()), columns=["Metric", "Value"]).applymap(
            lambda x: round(x, 2) if isinstance(x, (int, float)) else x
        )
        st.table(density_df.style.set_properties(**{"text-align": "center"}).set_table_styles([
            {"selector": "thead", "props": [("font-weight", "bold"), ("background-color", "#f7f7f7")]}
        ]))


if __name__ == "__main__":
    display_other_analysis()
