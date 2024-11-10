# app.py

import sys
import os

# Add the root directory to sys.path to allow importing from src as a module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now import the data_analysis module from src/pages
from src.pages import data_analysis  # Import the data_analysis page module
import streamlit as st

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "Data Analysis", "Model", "Evaluation and Visualization"])

# Display page based on selection
if page == "Home":
    st.title("Welcome to the Data Analysis App")
    st.write("""
    This app allows you to analyze data, build models, and evaluate results.
    Select an option from the sidebar to get started.
    """)
elif page == "Data Analysis":
    data_analysis.display_data_analysis()  # Call the function to display the Data Analysis page content
elif page == "Model":
    st.title("Model Page")
    st.write("Model training and evaluation coming soon.")
elif page == "Evaluation and Visualization":
    st.title("Evaluation and Visualization Page")
    st.write("Evaluation metrics and visualizations coming soon.")