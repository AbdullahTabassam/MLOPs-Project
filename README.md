# Enterprise Machine Learning
##	Overview

This file is used to explain the directory structure and contents of the files and directories used in the project in detail. This project can be run in two ways. After you clone the project, there will be two directories inside. 

```
Project/
├── Build_from_source/
└── Pull_from_hub/
```

## To Pull from Docker Hub 

In Pull_from_hub directory, there is only the 'saved_model' folder and a 'docker-compose.yml' file inside a sub-directory called 'enterprise_machine_learning'. 

```
enterprise_machine_learning
├── saved_model/
│   └── 1/
│       ├── saved_model.pb
│       └── variables/
│          ├── variables.data-00000-of-00001
│          └── variables.index
└── docker-compose.yml
```

Open a command prompt in the 'enterprise_machine_learning' folder inside 'Pull_from_hub' and run 'docker compose up' in case of compose V2 and if you have compose V1 run 'docker-compose up'. Docker will pull the tf2 serving image and flask app image from the hub and copy the saved model in the container volume specified in the compose file.
###### When you successfully build your image in the build from scratch part, you can upload it to Docker Hub and later put the name of your image in the composefile at the marked place to make it work. This step is an additional step to make the project more scalable reproducable. Morwe on uploading to docker <a href='https://www.techrepublic.com/article/how-to-build-a-docker-image-and-upload-it-to-docker-hub/'>here</a>.

## To Build from Scratch

To build from scratch, open command prompt in 'enterprise_machine_learning' folder inside 'Build_from_source' folder and run 'docker compose up' in case of compose V2 and if you have compose V1 run 'docker-compose up'.

In Build_from_source directory, all the source files are available. The complete project directory structure is shown below:

```
enterprise_machine_learning/
├── Web/
│   ├── FlaskObjectDetection/
│   │   ├── __pycache__/
│   │   ├── core/
│   │   ├── data/
│   │   │   └── mscoco_label_map.pbtxt
│   │   ├── protos/
│   │   ├── static/
│   │   │   ├── detection/
│   │   │   ├── uploads/
│   │   │   └── styles.css
│   │   ├── templates/
│   │   │   ├── About.html
│   │   │   ├── Contact.html
│   │   │   ├── Detection.html
│   │   │   ├── Feedback.html
│   │   │   ├── GetStarted.html
│   │   │   ├── Home.html
│   │   │   ├── Index.html
│   │   │   ├── Login.html
│   │   │   ├── Register.html
│   │   │   ├── SessionExpired.html
│   │   │   ├── template.html
│   │   │   └── ThankYou.html
│   │   ├── uploads/
│   │   ├── utils/
│   │   └── app.py
│   ├── saved_model/
│   │   └── 1/
│   │       ├── saved_model.pb
│   │       └── variables/
│   │          ├── variables.data-00000-of-00001
│   │          └── variables.index
│   ├── .dockerignore
│   └── dockerfile
└── docker-compose.yml
```

### Saved Model 

For this project we have used a pre-trained model from TF2 Model Zoo and placed it in the saved_model folder (enterprise_macine_learning/Web/saved_model). When creating the TF2 serving container, we will copy the contents of this folder to the /models/serving folder inside the TF2-serving docker container named DetectX. As it is not possible to upload the variables.data-00000-of-00001 file on gthub, you can download a model of you choice from <a href='https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md'>TF2 model zoo</a>. 

```
saved_model/
└── 1/
    ├── saved_model.pb
    └── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```

### Label maps: 

In the data folder (enterprise_macine_learning/Web/FlaskObjectDetection/data) we have placed the 'mscoco_label_map.pbtxt' file that contains the label maps for 90 different classes this model can detect.

```
data/
└── mscoco_label_map.pbtxt
```

### Flask Application 

The flask app provides all the configuration setting and routes to the webpages that are used in this app. The python code for the app is present in the 'app.py' file in 'FlaskObjectDetection' folder.

The 'templates' folder contain the HTML files for the webpages used. All these webages are based on a base html file called 'template.html'. Jinja 2 formating is used to write the html files.

The the styles used in the Html files are based on the CSS file called 'styles.css' present in the static folder, in addition to the 'detection' and 'uploads' folders. The files that the user upload are saved in 'uploads' folder and the classified images with detection boxes on them are saved in the 'detection' folder

```
FlaskObjectDetection/
├── static/
│   ├── detection/
│   │   └── XYZ.jpg
│   ├── uploads/
│   │   └── XYZ.jpg
│   └── styles.css
├── templates/
│   ├── About.html
│   ├── Contact.html
│   ├── Detection.html
│   ├── Feedback.html
│   ├── GetStarted.html
│   ├── Home.html
│   ├── Index.html
│   ├── Login.html
│   ├── Register.html
│   ├── SessionExpired.html
│   ├── template.html
│   └── ThankYou.html
└── app.py
```

### Database 

The database is created using MySQL and contains three tables; 'users', 'detections', and 'feedback'. The 'users' table is used for login and whenever a new user registers, thier information is stored in the users table. A user remains logged in unless they logout themselves or their session key expires. When the user makes a detection, all the stats related to that detection gets stored in the 'detections' table. We could have stored the images in the database also, but it is not professional and is against the best practices. Instead, the image name is stored in the database, as we are storing the images locally in the 'uploads' and 'detection' folders and can refernce the image, if needed, from these folders. Similarly, if a user leaves a feedback, it is stored in the 'feddback' table. 


### Docker file 

The dockerfile contains all the information for the build process of a docker container. We are using the dockerfile to containerise the flask application. In the docker file we have used a multi-stage build technique as we want the size of the container to be as small as possible. In the build-stage, we use a python:3.7 base image and first set up the working directory and copy the application files. Then, we install the dependencies and packages required by the application. This stage will generate a large intermediate image.
In the runtime-stage, we use the same version of the Python image to reduce the final image size. We copy the installed packages and the application files from the build-stage using the COPY command. Finally, we set the WORKDIR and CMD instructions to run the application.
We do not need any OS base image for this taska as the python:3 image that we are using in both stages already includes a minimal Linux distribution that is sufficient to run Python applications, so we can omit the ubuntu image altogether.


### Docker ignore file

We use a .dockerignore file so that when we build our images, we do not add unnecessary files that have no use in the container and just add up to size of the container. As a good practice such type of files are required to be ignored and can be done by calling such files in .dockerignore file.


### Docker Compose file

We can use docker compose file to pull or build images all at the same time without using the command lines. After the image has been built, we can push it to the Docker Hub and use a compose file pull the TF-serving and Flask app images using a single command of 'docker compose up'.


### Screenshots

We have also provided screenshots for the application created. In the screenshots folder, you can also observe the database and docker interface showing the size of the docker image. 


## Installation and requirements

To run this project you need to have Docker and Docker-compose installed in your device.
- Learn more about <a href='https://www.docker.com/'>docker</a> and <a href='https://docs.docker.com/compose/install/'>docker-compose</a>.

To clone this repository use this command:

    git clone https://github.com/AbdullahTabassam/MLOPs-Project