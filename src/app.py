# src/app.py

import sys
import os
import json
import streamlit as st

# Add the root directory to sys.path to allow importing from src as a module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

# Import configuration and required modules
from src.config import settings
from src.pages.data_analysis.compute_metrics import compute_and_save_metrics
from src.pages.data_analysis.data_analysis import display_data_analysis

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Data Analysis", "Model", "Evaluation and Visualization"])

# Display page based on selection
if page == "Home":
    st.title("Welcome to the Data Analysis App")
    st.write(
        """
        This app allows you to analyze data, build models, and evaluate results.
        Select an option from the sidebar to get started.
        """
    )

    # Check if metrics.json exists, and compute metrics if not present
    metrics_path = settings.METRICS_PATH
    if not os.path.exists(metrics_path):
        st.info("Metrics file not found. Computing metrics for the first time...")
        metrics = compute_and_save_metrics()
        st.success("Metrics computed and saved successfully.")
    else:
        # Load existing metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # Button to recompute metrics
    if st.button("Recompute Metrics"):
        metrics = compute_and_save_metrics()
        st.success("Metrics recomputed successfully.")

elif page == "Data Analysis":
    st.title("Data Analysis Page")
    display_data_analysis()

elif page == "Model":
    st.title("Model Page")
    st.write("Model training and evaluation coming soon.")

elif page == "Evaluation and Visualization":
    st.title("Evaluation and Visualization Page")
    st.write("Evaluation metrics and visualizations coming soon.")
