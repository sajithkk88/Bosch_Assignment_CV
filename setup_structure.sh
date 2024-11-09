#!/bin/bash

# Create folders
mkdir -p assignment_data_bdd
mkdir -p data_analysis
mkdir -p model
mkdir -p evaluation
mkdir -p docker
mkdir -p tests

# Create placeholder files in data_analysis folder
touch data_analysis/analysis.py
touch data_analysis/visualization.py
touch data_analysis/dashboard_app.py
touch data_analysis/utils.py
touch data_analysis/requirements.txt
touch data_analysis/__init__.py

# Create placeholder files in model folder
touch model/model.py
touch model/train.py
touch model/dataloader.py
touch model/utils.py
touch model/requirements.txt
touch model/__init__.py

# Create placeholder files in evaluation folder
touch evaluation/evaluate.py
touch evaluation/metrics.py
touch evaluation/visualization.py
touch evaluation/suggestions.md
touch evaluation/requirements.txt
touch evaluation/__init__.py

# Create placeholder files in docker folder
touch docker/Dockerfile
touch docker/docker-compose.yml
touch docker/README.md

# Create placeholder files in tests folder
touch tests/test_analysis.py
touch tests/test_model.py
touch tests/test_evaluation.py
touch tests/__init__.py

# Create root-level files
touch README.md
touch .gitignore

# Add entries to .gitignore
echo "assignment_data_bdd/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
echo ".ipynb_checkpoints" >> .gitignore

echo "Repository structure created successfully!"
