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
    """
    Display the Home Page with an introduction to the assignment solution.
    """

    st.title("Bosch Computer Vision Assignment")
    st.markdown("""
    ### Welcome to the Solution Portal
    This application is designed to provide an interactive overview of the Bosch Computer Vision assignment solution. 
    The features of this app include:
    - **Data Analysis**: Explore insights, visualizations, and summaries of the BDD100K dataset used for training.
    - **Model Training and Evaluation**: Detailed results and metrics from the YOLOv5 model retraining.
    - **Documentation**: Comprehensive details of the solution approach, methodologies, and improvements.

    ---
    
    #### How to Navigate
    Use the sidebar to explore:
    - **Data Analysis**: Dive into dataset characteristics, class distributions, and bounding box analyses.
    - **Model Details**: View training parameters, architecture, and performance metrics.
    - **Documentation**: Understand the solution's thought process, methodologies, and suggestions for improvement.

    ---
    #### About the Solution
    - **Dataset**: The BDD100K dataset, tailored for autonomous driving research.
    - **Model**: YOLOv5 architecture, fine-tuned for object detection.
    - **Tools**: Python, Streamlit, Docker, and other state-of-the-art libraries and frameworks.

    Explore the app to gain insights into the entire workflow, from data preprocessing to model evaluation.
    """)

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

elif page == "Data Analysis":
    st.title("Data Analysis Page")
    display_data_analysis()

elif page == "Model":
    st.title("Model Page")
    display_model_details()
    
elif page == "Documentation":
    st.title("Documentation")
    display_documentation()

