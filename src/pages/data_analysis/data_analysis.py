# src/pages/data_analysis.py

import streamlit as st
from src.pages.data_analysis import dataset_overview, class_distribution, other_analysis

def display_data_analysis():
    """Display the Data Analysis page with multiple dropdown options."""
    #st.title("Data Analysis")
    #st.write("This page provides insights and analysis on the dataset.")
    
    # Dropdown menu for analysis options
    analysis_option = st.selectbox("Select Analysis Type", 
                                   ["Dataset Overview", "Class Distribution", "Other Analysis"])
    
    # Display the appropriate analysis based on user selection
    if analysis_option == "Dataset Overview":
        dataset_overview.display_dataset_overview()
    elif analysis_option == "Class Distribution":
         class_distribution.display_class_distribution()
    elif analysis_option == "Other Analysis":
        other_analysis.display_other_analysis()
