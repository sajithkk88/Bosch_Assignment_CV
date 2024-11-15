To run the app from repo:

In Ubuntu run from Bash, IN windows run from Ubntu app or VS code terminal
git clone https://github.com/sajithkk88/Bosch_Assignment_CV
pip install -r requirements.txt
streamlit run .\src\app.py --> this should open the web app in the default bowser, if not paste the URL http://localhost:8501/

To run the app using docker image (assume docker is installed and docker desktop is running)

1. Download the docker image 'bosch-cv-assignment-app.tar'
2. in Ubuntu open bash, in Windows open cmd
3. Change the directory to where the Docker image file
4. docker load < bosch-cv-assignment-app.tar
5. find the absolute path of folder having BDD100k data with folders 'bdd100k_images_100k' and 'bdd100k_labels_release'.
6. docker run -it --rm -p 8501:8501 -v path\to\local\data:/app/data/assignment_data_bdd bosch-cv-assignment-app
7. open a web browser and paste http://localhost:8501 -> this will open the app 


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
