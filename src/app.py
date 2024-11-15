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
from src.pages.model.display_model_details import display_model_details
from src.pages.documentation.document_details import display_documentation


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Data Analysis", "Model","Documentation"])

# Display page based on selection
if page == "Home":
    st.title("Bosch Applied CV Coding Assignment")
    st.write(
        """
        To selectively view different aspects of the solution of assignment.
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
    display_model_details()
    
elif page == "Documentation":
    st.title("Documentation")
    display_documentation()

