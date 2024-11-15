# create_bdd100k_yaml.py

import os

def create_bdd100k_yaml(file_path="src/pages/model/yolov5/data/bdd100k.yaml"):
    # Define YAML content
    yaml_content = """train: data/images/train  # Path to training images
val: data/images/val      # Path to validation images

# Define class names for BDD100K 
names:
  0: pedestrian
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
  8: traffic light
  9: traffic sign
"""

    # Write to file
    with open(file_path, "w") as file:
        file.write(yaml_content)
    print(f"{file_path} created successfully.")

# Run the function to create the YAML file
if __name__ == "__main__":
    create_bdd100k_yaml()
