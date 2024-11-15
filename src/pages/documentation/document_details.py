# src/pages/documentation/document_details.py

import streamlit as st
import pandas as pd
from PIL import Image

def display_documentation():
    """
    Display the Documentation page with multiple sections.
    Provides a structured template for documenting the assignment.
    """
    # Dropdown menu for documentation sections
    documentation_section = st.selectbox(
        "Select Documentation Section",
        [
            "Introduction",
            "Data Analysis",
            "Model Development",
            "Evaluation and Visualization",
            "Usage Instructions"
        ]
    )

    # Display the appropriate section based on user selection
    if documentation_section == "Introduction":
        st.markdown("### Introduction")
        st.write("""
        This assignment focuses on the end-to-end pipeline for object detection using the BDD dataset. 
        The tasks include data analysis, model training, evaluation, and visualization, with documentation for 
        reproducibility and deployment.
        """)
        st.write("""
        Key tasks include:
        - Analyzing the dataset for insights and patterns.
        - Training a pre-trained object detection model.
        - Evaluating the model and providing detailed visualizations.
        - Ensuring deployment readiness via containers.
        """)

    elif documentation_section == "Data Analysis":
        st.markdown("### Data Analysis")
        st.write("""
        The data analysis section covers:
        - Parsing and inspecting the dataset.
        - Distribution of object detection classes in train and val datasets.
        - Identifying anomalies or unique samples.
        - Visualizing statistics and patterns via dashboards.
        """)
        
        # Dataset Split Documentation
        st.markdown("### Dataset Split")
        st.write("""
        The dataset is divided into training, validation, and test sets:
        - **Train**: 70,000 images (70%)
        - **Validation**: 20,000 images (20%)
        - **Test**: 10,000 images (10%)
        """)

        # Class Distribution Documentation
        st.markdown("### Class Distribution")
        st.write("This section displays the distribution of classes within the training and validation datasets.")

        # Training Class Distribution
        st.markdown("#### Training Set Class Distribution")
        st.image("src/pages/documentation/images/class_distribution_train.png", caption="Class Distribution - Training")
        st.write("""
        - The training dataset has a skewed distribution with the majority of instances belonging to the 'car' class.
        - 'Traffic light' and 'traffic sign' classes also have a significant presence.
        - Classes like 'rider', 'train', 'bus', and 'truck' are underrepresented, indicating an imbalanced dataset.
        """)

        # Validation Class Distribution
        st.markdown("#### Validation Set Class Distribution")
        st.image("src/pages/documentation/images/class_distribution_val.png", caption="Class Distribution - Validation")
        st.write("""
        - Similar to the training set, the validation dataset shows a heavy skew towards the 'car' class.
        - 'Traffic light' and 'traffic sign' maintain a prominent count, while other classes remain underrepresented.
        """)

        # Bounding Box Analysis Documentation
        st.markdown("### Bounding Box Analysis")
        st.write("The bounding box analysis includes area and aspect ratio statistics for objects in the dataset.")

        # Training Bounding Box Summary
        st.markdown("#### Training Set Bounding Box Summary")
        st.image("src/pages/documentation/images/bounding_box_summary_train.png", caption="Bounding Box Summary - Training")
        st.write("""
        - 'Car' class exhibits the largest bounding box areas on average, followed by 'bus' and 'truck'.
        - 'Train' has high aspect ratios, indicating elongated shapes, while 'rider' and 'traffic light' have smaller aspect ratios.
        - Variance in bounding box area and aspect ratio suggests diversity in object sizes within classes.
        """)

        # Validation Bounding Box Summary
        st.markdown("#### Validation Set Bounding Box Summary")
        st.image("src/pages/documentation/images/bounding_box_summary_val.png", caption="Bounding Box Summary - Validation")
        st.write("""
        - Bounding box statistics for the validation set are consistent with the training set.
        - The 'car' class has the largest bounding box areas, and high aspect ratios are observed for 'train'.
        """)

        # Scene Context Analysis Documentation
        st.markdown("### Scene Context Analysis")
        st.write("The scene context analysis details object distributions across different scene types.")

        # Training Scene Context
        st.markdown("#### Training Set Scene Context")
        st.image("src/pages/documentation/images/scene_context_analysis_train.png", caption="Scene Context - Training")
        st.write("""
        - Most objects in the training set are present in 'city street' and 'highway' scenes.
        - 'Cars' dominate the 'city street' and 'highway' scenes, while other classes are relatively sparse.
        - Classes like 'train' and 'rider' are infrequent across all scene types.
        """)

        # Validation Scene Context
        st.markdown("#### Validation Set Scene Context")
        st.image("src/pages/documentation/images/scene_context_analysis_val.png", caption="Scene Context - Validation")
        st.write("""
        - Similar scene distribution trends are observed in the validation set.
        - The 'city street' scene has the highest count of objects, dominated by 'car', 'traffic sign', and 'traffic light' classes.
        """)
        
        st.markdown("### Object Attributes")
        st.subheader("Object Attributes - Train")
        st.image("src/pages/documentation/images/object_attributes_train.png", caption="Train - Object Attributes", use_column_width=True)
        st.write("""
        - **Occlusion Counts:** High occlusion counts for `car`, `traffic sign`, and `bus` classes suggest that a significant number of objects are partially visible. This can help models become robust to partially visible objects.
        - **Truncation Counts:** Many instances of `car` and `traffic sign` are truncated due to image boundaries, which reflects the real-world scenario of large or nearby objects being only partially visible.
        - **Traffic Light Colors:** Distribution shows `green` and `red` as the most common states, aligning with typical traffic light usage.
        """)
        
        st.subheader("Object Attributes - Validation")
        st.image("src/pages/documentation/images/object_attributes_val.png", caption="Validation - Object Attributes", use_column_width=True)
        st.write("""
        - Similar trends to the training set, with `car` and `traffic sign` showing high occlusion and truncation counts.
        - The consistency between training and validation sets in occlusion, truncation, and traffic light colors ensures that the model will encounter similar attribute distributions in both sets.
        """)
        
        st.markdown("### Object Density")
        st.subheader("Object Density - Train")
        st.image("src/pages/documentation/images/object_density_train.png", caption="Train - Object Density", use_column_width=True)
        st.write("""
        - The average number of objects per image is around 27.8, with a moderate standard deviation, indicating a good mix of crowded and sparse scenes.
        - The maximum of 107 objects per image highlights the presence of highly dense scenes, which could challenge the model's detection capabilities.
        """)
        
        st.subheader("Object Density - Validation")
        st.image("src/pages/documentation/images/object_density_val.png", caption="Validation - Object Density", use_column_width=True)
        st.write("""
        - Similar density statistics as the training set, with an average of around 27.9 objects per image.
        - Consistent object density across both sets should enable the model to generalize well to varying scene complexities.
        """)

    elif documentation_section == "Model Development":
        st.markdown("### Model Development")
        st.write("""
        This section discusses the chosen model for the task:
        - **Model Name**: YOLOv5 
        - **Reasoning**: YOLOv5 is known for its efficiency and accuracy in real-time object detection tasks. 
        Comparison with other popular object detection models like SSD and Faster R-CNN are given in the table below.
        Real time detection is of prime importance since application is for autonomous driving.
        Version 5 is chosen because it was released in 2020 and there are more documents available explaining the architecture and steps for training. 
        The capability of the pretrai
        It supports a wide range of pre-trained weights for transfer learning.
        - **Custom Model**: It is pretrained to detect 80 classes on COCO dataset. I retrained it on the BDD100k dataset to do object detection of 10 classes.
        """)
        data = {
        "Criteria": [
        "Speed", "Accuracy", "Latency", "Real-Time Capability",
        "Object Size Handling", "Robustness in Scenarios",
        "Ease of Deployment", "Training Complexity",
        "Multi-Class Detection", "Community Support"
        ],
        "YOLO": [
        "Real-time (high FPS, suitable for onboard systems)",
        "Balanced accuracy for most objects; struggles with very small ones",
        "Low latency (single forward pass)",
        "Excellent (suitable for autonomous driving)",
        "Improved in later versions (YOLOv4, v5, v7)",
        "Robust in dynamic, complex environments (real-world driving)",
        "Lightweight, deployable on edge devices (e.g., GPUs in vehicles)",
        "Simple, with available pre-trained weights and fine-tuning options",
        "Efficient for multi-class detection in driving (cars, pedestrians)",
        "Large community with frequent updates (YOLOv5, YOLOv8, etc.)"
        ],
        "SSD": [
        "Fast but slower than YOLO, especially at higher resolutions",
        "Better for small objects due to multi-layer feature maps",
        "Moderate latency (multi-layer processing)",
        "Good but less optimized for real-time",
        "Good for small and large objects due to multi-scale layers",
        "Moderate robustness; struggles in highly dynamic settings",
        "Flexible but requires more computational resources",
        "Moderate complexity; flexible backbone choices",
        "Also efficient; supports varied datasets",
        "Good community support; versatile backbones like MobileNet"
        ],
        "Faster R-CNN": [
        "Slow (two-stage detector, not suitable for real-time use)",
        "High accuracy but prone to higher latency",
        "High latency (multi-stage process)",
        "Poor for real-time; better for static or slow-moving scenes",
        "Very good for all object sizes but at the cost of speed",
        "Robust but slow, limiting real-world responsiveness",
        "Computationally expensive; best suited for server deployment",
        "High complexity; requires significant computational power",
        "Effective but slower for multi-class detection",
        "Strong but niche; used in high-accuracy research contexts"
        ]
        }
        
        data_df = pd.DataFrame(data)
        st.table(data_df) 
        
        st.write("""
        Key steps in the model pipeline:
        - Yolov5 github repo was cloned first and necessary requirements were installed.
        -  Next step was to convert the BDD100k dataset to a dataset conforming to Yolo format. A custom script does this.
        - A .yaml file was created using a custom script to indicate that only 10 classes need to be detected.
        - Training was done using the script provided in the git repo.
        - Fine-tuning and evaluation on the validation dataset.
        - Details of the retraining can be viewed in the page Model>Training Details
        """)

    elif documentation_section == "Evaluation and Visualization":
        """
        Display the model evaluation metrics and insights on the page.
        This includes final metric values and graphical analysis of confusion matrix, 
        F1-Confidence Curve, Precision-Confidence, Recall-Confidence, and Precision-Recall Curve.
        """

        # Section 1: Final Metric Values
        st.header("Final Evaluation Metrics")
        st.write("""
        The following metrics reflect the overall performance of the model on the validation dataset.
        - **Mean Average Precision (mAP@0.5):** 0.504
        - **Precision:** 0.68
        - **Recall:** 0.67
        - **F1 Score:** 0.52
        """)

        # Insights
        st.markdown("### Insights")
        st.write("""
        - The mean average precision (mAP@0.5) of 0.504 shows moderate detection accuracy across all classes.
        - Precision and recall values indicate a reasonable balance between the model’s ability to avoid false positives and capture true positives.
        - The F1 Score is relatively lower for rare classes like 'train' and 'rider', reflecting class imbalance.
        """)

        # Section 2: Confusion Matrix
        st.header("Confusion Matrix")
        confusion_img = Image.open("src/pages/model/training_bdd100k_finetune_epoch50_batch16/confusion_matrix.png")
        st.image(confusion_img, caption="Confusion Matrix", use_container_width=True)
    
        st.write("""
        - The confusion matrix highlights the high accuracy in identifying the `car` class, which has ample representation in the training data.
        - Classes like `truck` and `bus` have notable confusion, which could be due to visual similarity.
        - Classes with low sample representation, such as `train` and `rider`, show poor classification accuracy, reinforcing the impact of class imbalance.
        """)

        # Section 3: F1-Confidence Curve
        st.header("F1-Confidence Curve")
        f1_curve_img = Image.open("src/pages/model/training_bdd100k_finetune_epoch50_batch16/F1_curve.png")
        st.image(f1_curve_img, caption="F1-Confidence Curve", use_container_width=True)
    
        st.write("""
        - Classes such as `car` and `traffic sign` achieve relatively higher F1 scores across confidence thresholds, reflecting strong performance on common objects.
        - Lower F1 scores for `train` and `rider` highlight the difficulty in classifying underrepresented classes, suggesting possible improvements through data augmentation or fine-tuning.
        """)

        # Section 4: Precision-Confidence and Recall-Confidence Curves
        st.header("Precision-Confidence Curve")
        p_curve_img = Image.open("src/pages/model/training_bdd100k_finetune_epoch50_batch16/P_curve.png")
        st.image(p_curve_img, caption="Precision-Confidence Curve", use_container_width=True)

        st.header("Recall-Confidence Curve")
        r_curve_img = Image.open("src/pages/model/training_bdd100k_finetune_epoch50_batch16/R_curve.png")
        st.image(r_curve_img, caption="Recall-Confidence Curve", use_container_width=True)
    
        st.write("""
        - The `car` class achieves high precision and recall across confidence levels, affirming its dominance in the dataset.
        - Classes with fewer instances, like `train`, exhibit lower precision and recall, suggesting a need for more training samples or adjusted thresholds.
        """)

        # Section 5: Precision-Recall Curve
        st.header("Precision-Recall Curve")
        pr_curve_img = Image.open("src/pages/model/training_bdd100k_finetune_epoch50_batch16/PR_curve.png")
        st.image(pr_curve_img, caption="Precision-Recall Curve", use_container_width=True)
    
        st.write("""
        - The Precision-Recall curve indicates high precision for `car`, `traffic sign`, and `traffic light` classes, while rare classes suffer from lower precision-recall trade-offs.
        - Additional training data for underrepresented classes could improve recall without sacrificing precision, enhancing overall performance.
        """)

        # Section 6: Training and Validation Loss Curves
        st.header("Training and Validation Loss Curves")
        results_img = Image.open("src/pages/model/training_bdd100k_finetune_epoch50_batch16/results.png")
        st.image(results_img, caption="Loss and Performance Metrics over Epochs", use_container_width=True)
    
        st.write("""
        - The gradual reduction in training and validation losses indicates effective learning and convergence of the model.
        - The validation mAP stabilizes, suggesting that further training may have diminishing returns, or it could be improved by modifying the model architecture or loss function.
        - Higher metrics for the frequently occurring `car` and `traffic sign` classes reaffirm the importance of balanced class distribution for robust model performance.
        """)
        
        # Section 7: Qualitative Analysis
        st.header("**Qualitative Analysis**")
        
        st.write("""
        - Inference result of 20 images from test data (which model has not seen before) are given in the page Model > Inferencing on Test Data.
        - It is observed that carsvare detected with mostly good confidence in day ans well as night.
        - Traffic signs are detected with less to moderate confidence.
        """)
        
        # Section 8: Suggestions for Improvement
        st.header("**Suggestions for Improvement**")

        st.write("""
        Based on the performance metrics and analysis, the following improvements are suggested to enhance model accuracy and robustness:

        1. **More Data for Underrepresented Classes**:
        - Gathering more data samples for rare classes like `train`, `rider`, and `bus` could significantly improve the recall and precision for these categories.

        2. **Experiment with Model Architectures**:
        - Exploring alternative model architectures (e.g., YOLOv6, YOLOv7, or Transformer-based models) could improve accuracy and efficiency.
        
        3. ** Train with More Epochs**
        - More epochs could atleast help it converge to a slightly better value.
        """)

    elif documentation_section == "Usage Instructions":
        st.markdown("### Usage Instructions")
        st.write("""
        To do retrainign of yolov5 model:
        1. Clone the repository: https://github.com/ultralytics/yolov5 and https://github.com/sajithkk88/Bosch_Assignment_CV
        2. Ensure all dependencies are installed (`requirements.txt` provided).
        3. Copy scripts named generate_bdd100k_yaml_file.py, convert_bdd_to_yolo_format.py and test_model_on_test_images to root folder of yolov5 repo.
        4. Edit convert_bdd_to_yolo_format.py and change DATA_PATH to the absolute pathof folder having BDD100k data with folders 'bdd100k_images_100k' and 'bdd100k_labels_release'.
        5. from root folder of yolv5 repo, run 'convert_bdd_to_yolo_format.py'. This will convert the images and labels of BDD100k data to yolo format and save in /data/images and /data/labels respectively 
        6. from root folder run generate_bdd100k_yaml_file.py to generate bdd100k.yaml in /data folder which has 10 classes information.
        7. Run train command with multiple option: eg., python train.py --weights yolov5s.pt --data data/bdd100k.yaml --img 640 --epochs 50 --batch-size 16 --name bdd100k_finetune --device 0
        8. Tun val command to evaluate the model: eg., python val.py --weights yolov5s.pt --data data/bdd100k.yaml --img 640 --task val
        9. To do inferencing on a small set of test images run test_model_on_test_images
        
        To run the app from repo:

        In Ubuntu run from Bash, IN windows run from Ubntu app or VS code terminal
        1. git clone https://github.com/sajithkk88/Bosch_Assignment_CV
        2. pip install -r requirements.txt
        3. streamlit run .\src\app.py --> this should open the web app in the default bowser, if not paste the URL http://localhost:8501/

        To run the app using docker image (assume docker is installed and docker desktop is running)

        1. Download the docker image 'bosch-cv-assignment-app.tar'
        2. in Ubuntu open bash, in Windows open cmd
        3. Change the directory to where the Docker image file
        4. docker load < bosch-cv-assignment-app.tar
        5. find the absolute path of folder having BDD100k data with folders 'bdd100k_images_100k' and 'bdd100k_labels_release'.
        6. docker run -it --rm -p 8501:8501 -v path\to\local\data:/app/data/assignment_data_bdd bosch-cv-assignment-app
        7. open a web browser and paste http://localhost:8501 -> this will open the app 

        """)

if __name__ == "__main__":
    display_documentation()
