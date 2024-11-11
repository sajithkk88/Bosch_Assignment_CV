import os

# Base path for data (overridable by an environment variable)
DATA_PATH = os.getenv("DATA_PATH", "data/assignment_data_bdd")

# Paths for label JSON files
LABELS_PATH = os.path.join(DATA_PATH, "bdd100k_labels_release/bdd100k/labels")
TRAIN_LABELS_PATH = os.path.join(LABELS_PATH, "bdd100k_labels_images_train.json")
VAL_LABELS_PATH = os.path.join(LABELS_PATH, "bdd100k_labels_images_val.json")

# Base path for images
IMAGES_PATH = os.path.join(DATA_PATH, "bdd100k_images_100k/bdd100k/images/100k")
TRAIN_IMAGES_PATH = os.path.join(IMAGES_PATH, "train")
TEST_IMAGES_PATH = os.path.join(IMAGES_PATH, "test")
VAL_IMAGES_PATH = os.path.join(IMAGES_PATH, "val")

# Base data directory for computed metrics
DATA_ANALYSIS_PATH = os.path.join("src", "pages", "data_analysis")
METRICS_PATH = os.path.join(DATA_ANALYSIS_PATH, "metrics.json")
