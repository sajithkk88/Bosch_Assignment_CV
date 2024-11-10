import os

# Define the folder structure
FOLDER_STRUCTURE = {
    "data": [
        "README.md"  # Instructions on where to place data
    ],
    "src": {
        "app.py": None,  # Main entry point for the Streamlit app
        "pages": [
            "data_analysis.py",  # Code for 'Data Analysis' page
            "model.py",          # Code for 'Model' page
            "evaluation.py"      # Code for 'Evaluation and Visualization' page
        ],
        "utils": [
            "data_loader.py",           # Functions for loading data
            "analysis_helpers.py",      # Helper functions for data analysis
            "model_helpers.py",         # Helper functions for model operations
            "visualization_helpers.py"  # Helper functions for visualization
        ],
        "config": [
            "settings.py"  # Global settings and configurations (e.g., paths, parameters)
        ]
    },
    "Dockerfile": None,  # Dockerfile to build the Docker image
    "requirements.txt": None,  # Python dependencies for the project
    ".dockerignore": None,  # Files and folders to exclude from the Docker image
    ".gitignore": None,  # Files and folders to exclude from Git version control
    "README.md": None  # Project overview and instructions
}

def create_structure(base_path, structure):
    """
    Recursively creates the folder structure based on the given dictionary.
    :param base_path: The base path where folders and files will be created.
    :param structure: A dictionary defining folder and file structure.
    """
    for name, content in structure.items():
        path = os.path.join(base_path, name)

        if isinstance(content, dict):  # If content is a dictionary, create a folder
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")

            # Recursively create subfolders and files within this folder
            create_structure(path, content)
        elif isinstance(content, list):  # If content is a list, create files within a folder
            os.makedirs(path, exist_ok=True)
            print(f"Created folder: {path}")

            for file_name in content:
                file_path = os.path.join(path, file_name)
                open(file_path, 'w').close()
                print(f"Created file: {file_path}")
        else:  # If content is None, it's a single file to be created at the base level
            open(path, 'w').close()
            print(f"Created file: {path}")

# Set the base directory to the script's directory
base_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_root")
create_structure(base_directory, FOLDER_STRUCTURE)
