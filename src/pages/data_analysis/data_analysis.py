# src/pages/data_analysis/data_analysis.py

import streamlit as st
from src.pages.data_analysis import dataset_overview, class_distribution, other_analysis
from src.pages.data_analysis.compute_metrics import compute_and_save_metrics

def display_data_analysis():
    """
    Display the Data Analysis page with multiple dropdown options.
    Provides options to view Dataset Overview, Class Distribution, or Other Analysis.
    """
    # Button to recompute metrics
    st.write("Click the button to recompute metrics (metrics for data analysis are precomputed and saved for faster loading):")
    if st.button("Recompute Metrics"):
        metrics = compute_and_save_metrics()
        st.success("Metrics recomputed successfully.")
    
    # Dropdown menu for analysis options
    analysis_option = st.selectbox(
        "Select Analysis Type", 
        ["Dataset Overview", "Class Distribution", "Other Analysis"]
    )
    
    # Display the appropriate analysis based on user selection
    if analysis_option == "Dataset Overview":
        dataset_overview.display_dataset_overview()
    elif analysis_option == "Class Distribution":
        class_distribution.display_class_distribution()
    elif analysis_option == "Other Analysis":
        other_analysis.display_other_analysis()
