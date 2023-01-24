#Import important libraries

from flask import Flask, config, render_template, request, redirect, url_for, flash, session
from pandas import options
from werkzeug.utils import secure_filename
import numpy as np
import os
import sys
import tensorflow as tf
from PIL import Image
import pandas as pd
import grpc
from flask_mysqldb import MySQL

# Import prediction service functions from TF-Serving API
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2, get_model_metadata_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from utils import label_map_util
from utils import visualization_utils as viz_utils
from core.standard_fields import DetectionResultFields as dt_fields

sys.path.append("..")
tf.get_logger(). setLevel('ERROR')

# labels file for the 90 class model
PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt"
# number of classes to classify
NUM_CLASSES = 90

# maps the index number and category names of the classes
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# initailise Flask instance and configure parameters
app = Flask(__name__)

# To use flash message and keep user logged in (by creating a 24 character session key)
app.secret_key = os.urandom(24)

# configure uploads folder with the app
app.config['UPLOAD_FOLDER'] = 'uploads/'
# file extensions allowed to upload
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# MYSQL database config.  Goto https://www.phpmyadmin.co/ to view the database.
app.config['MYSQL_HOST'] = 'sql8.freemysqlhosting.net'
app.config['MYSQL_USER'] = 'sql8611078'
app.config['MYSQL_PASSWORD'] = 'Nca6eBAjsk'
app.config['MYSQL_DB'] = 'sql8611078'

mysql = MySQL(app)

# function to check for allowed image extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# this function creates interface to communicate with tensorflow serving
def get_stub(host='DetectX', port='8500'):   # Can not use container IP as it is Dynamic, instead used container name as host.
    channel = grpc.insecure_channel('DetectX:8500') # Port 8500 is mapped to port 8500 of the container.
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    return stub


# convert image into numpy array of the shape height, width, channel
def load_image_into_numpy_array(image):
    im_width, im_height = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# tensorlfow requires the input to be a tensor
# convert numpy array to tensorflow tensor
def load_input_tensor(input_image):
    image_np = load_image_into_numpy_array(input_image)
    image_np_expanded = np.expand_dims(image_np, axis=0).astype(np.uint8)
    tensor = tf.make_tensor_proto(image_np_expanded)
    return tensor


# function to perform inference on the image and draw bounding boxes
# it also returns the class names with their accuracy
def inference(frame, stub, model_name='serving'):
    # Add the RPC command here
    # Call tensorflow server
    # channel = grpc.insecure_channel('localhost:8500')
    channel = grpc.insecure_channel('DetectX:8500', options=(('grpc.enable_http_proxy',0),))
    print("Channel: ", channel)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    print('Stub: ', stub)
    request = predict_pb2.PredictRequest()
    print('Request: ', request)
    request.model_spec.name = 'serving'
    

    image = Image.fromarray(frame)
    input_tensor = load_input_tensor(image)
    request.inputs['input_tensor'].CopyFrom(input_tensor)

    result = stub.Predict(request, 60.0) # wait 1 min (60 sec) before request time out.

    # load image into numpy array
    image_np = load_image_into_numpy_array(image)
    # Copy of the original image_np is created and passed to the function
    # Both original and copy will be saved
    image_np_with_detections = image_np.copy()

    # the classes, bounding boxes, and accuracy scores are extracted 
    # and stored in a dictionary
    output_dict = {}
    output_dict['detection_classes'] = np.squeeze(
        result.outputs[dt_fields.detection_classes].float_val).astype(np.uint8)
    output_dict['detection_boxes'] = np.reshape(
        result.outputs[dt_fields.detection_boxes].float_val, (-1, 4))
    output_dict['detection_scores'] = np.squeeze(
        result.outputs[dt_fields.detection_scores].float_val)

    # method to draw bounding boxes on the image
    frame = viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,    # Max number of boxes that can be detcted is 200
                min_score_thresh=.80,     # Min accuracy threshold is set to 80%. Lower detections will be neglected.
                agnostic_mode=False)

    # the names of the classes along with their accuracy are extracted and 
    # stored in the variable detected_objects
    detected_objects = [(category_index.get(value)['name'], 
                        int(output_dict['detection_scores'][index]*100)/100) # converted to percentage for convenience
                        for index, value in enumerate(output_dict['detection_classes'])
                        if output_dict['detection_scores'][index] > 0.8]  # Min accuracy threshold is set to 80%. Lower detections will be neglected.
    
    return frame, image_np, detected_objects


# Declare the routes of the web app. 

# Route for homepage (logged out)
@app.route('/')
def index():
    if 'user_id' in session:                   # if session is acctive redirect to the logeed in Homepage.
        return redirect('/Home')
    else:
        return render_template('index.html')   # otherwise open the logged out homepage. 


# Route for Homepage (logged in)
@app.route('/Home')
def home():
    if 'user_id' in session:
        return render_template('Home.html')
    else:
        return redirect('/')

# Route for login page
@app.route('/Login', methods = ['POST', 'GET'])
def Login():
    if 'user_id' in session:            # if session is active
        return redirect('/Home')        # goto home page logged in
    else:                               # otherwise, check if the user information provided in form, matches to database.
        if request.method == 'POST':   

            Email = request.form['Email']
            password = request.form['password']
            cur = mysql.connection.cursor()
            #cur = connection.cursor()
            cur.execute( "SELECT * FROM users WHERE Email = %s AND password = %s",(Email,password) )
            user = cur.fetchall()
            
            if len(user)>0:
                session['user_id']=user[0][0]     # save the user id in the user id session
                session['user'] = user            # dave the complete info of user in user sesiion (to be used in other functions)
                return redirect(url_for('home'))  # Take the user to logged in homepage
            else:                                 # If details do not match the database
                flash('Enter correct Email and Password') # falsh this message
            return redirect(url_for('Login'))             # Show the login page again 
        return render_template('Login.html')

# Route for register page
@app.route('/Register', methods = ['POST', 'GET'])
def Register():
    if 'user_id' in session:                    # If session is active
        return redirect('/Home')                # goto logged in home page
    else:                                       # otherwise, take the info from register form
        if request.method == 'POST':
            First_name = request.form['First_name']
            Last_name = request.form['Last_name']
            age_verification = request.form['age_verification'] 
            Email1 = request.form['u_Email']
            Email2 = request.form['re_Email']
            password1 = request.form['u_password']
            password2 = request.form['re_password']

            if Email1==Email2 and password1==password2:  # Check for humman error in spellings
                Email = Email1
                password = password1
                cur = mysql.connection.cursor()         # activate cursor
                # push the data to users data base if the data does not already exsisit 
                cur.execute( "SELECT * FROM users WHERE Email = %s AND password = %s",(Email,password) )
                user = cur.fetchall()
                if len(user)>0:  # if already exsist
                    flash('Email ID already exsist. Login instead.') # flash this message
                    return redirect(url_for('Register'))             # and redirect to register page
                else:
                    cur.execute( "INSERT INTO users (First_name,Last_name,age_verification,Email,password) VALUES('{}', '{}', '{}', '{}', '{}')".format(First_name,Last_name,age_verification,Email,password))
                    mysql.connection.commit() # commit the change
                    cur.close()               # close the cursor
                    return redirect('/Login')
            else:
                flash('Email or Password do not match.') # if spelling check fails
                return redirect(url_for('Register'))     # flash this message
        return render_template('Register.html')


# Route for feedback page
@app.route('/Feedback', methods = ['POST', 'GET'])
def Feedback():
    if 'user_id' in session:       # if session is active, post the user info and submitted feedback to the feedbcak table in database
        if request.method == 'POST':
            user = session.get('user', None)
            user_id = user[0][0]
            First_name = user[0][1]
            Last_name = user[0][2]
            Email = user[0][4]
            print(First_name)
            feedback = request.form['feedback']   # get the feedback from form
            
            cur = mysql.connection.cursor()
            cur.execute( "INSERT INTO feedback (user_id,First_name,Last_name,Email,feedback) VALUES('{}','{}','{}','{}','{}')".format(user_id,First_name,Last_name,Email,feedback))
            mysql.connection.commit()  # Commit the change
            
            cur.close()                 # close the cursor
            return redirect(url_for('ThankYou'))  # Go to thanku=you page after feedback is uploaded
        return render_template('Feedback.html')
    else:                            # if there is no session,
        return redirect('/Login')    # do not let the user go to feed back page manually, and always take them to login page

# Route for logout
@app.route('/Logout')
def Logout():
    if 'user_id' in session:        # if session is active
        session.pop('user_id')      # logout the session and
        return redirect('/Login')   # take back to login page
    else:                           # otherwise
        return redirect('/Login')   # Go straight to login page

# Route for contact us page
@app.route('/Contact')
def Contact():
    return render_template('Contact.html')

# Route for about us page
@app.route('/About')
def About():
    return render_template('About.html')

# Route for
@app.route('/ThankYou')
def ThankYou():
    if 'user_id' in session:           # when feedback is submitted after logging in,  
        return redirect('/ThankYou')   # show thanks page 
    else:                              # but if there is no session 
        return redirect('/')           # redirect to unlogged home page

# Route for Getstarted page
@app.route('/GetStarted')
def GetStarted():
    if 'user_id' in session:   #if session is active go to getstarted page else goto login page
        return render_template('GetStarted.html')
    else:
        return redirect('/Login')


# Route for session expired page
@app.route('/401')
def SessionExpired():  
    return render_template('SessionExpired.html')


# Route for uploading the imgae
@app.route('/upload', methods=['GET','POST'])   # the methods supported for this route are both upload and post
def upload():
    if request.method == 'POST':
    # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')   # flash the message
            return redirect(request.url)

        file = request.files['file']
        # if no file is selected 
        if file.filename == '':
            flash('No file selected') # flash the message
            return redirect(request.url)
        # if the file is of correct extention show the result page
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('result',filename=filename))
        else: #if the file extention is not correct,
            flash('Please select an allowed file type (*.jpg, *.jpeg, *.png)') #flash this message
            return redirect(request.url)
        
    else:       # If method is not post go to getstarted page if logged in (otherwise it will take you to login page(line 278)) 
        return redirect(url_for('GetStarted'))   

# Route for Detections page.
@app.route('/Detection/<filename>')
def result(filename):
    if 'user_id' in session:                        # if the session is active, pull the user info from the session
        user = session.get('user', None)
        user_id = user[0][0]
        First_name = user[0][1]
        Last_name = user[0][2]
        Email = user[0][4]
        detected_obj = uploaded_file(filename)
        detected_obj_db = [(item[0], item[1]*100) for item in detected_obj]  # list containing detections
        cur = mysql.connection.cursor()             # Cursor activated
        
        for item in detected_obj_db: # Instert the detections and user info to the 'detections' table
            cur.execute("INSERT INTO detections (user_id,First_name,Last_name,Email,filename,object,confidence) VALUES('{}','{}','{}','{}','{}','{}','{}')".format(user_id,First_name,Last_name,Email,filename,item[0],(item[1])))
            mysql.connection.commit()  # Commit the change
            
        cur.close()                    # Close the cursor
        # detected objects sets is converted to dataframe
        df = pd.DataFrame(detected_obj_db, columns=['Objects', 'Accuracy'])
        # dataframe is converted to html to be displayed on the website
        df_html = df.to_html()
        # return the result html file
        return render_template('Detection.html', filename=filename, dataframe=df_html)   #Show the detections page
    else:
        return redirect('/401')         # if the session has expired show session expired page
def uploaded_file(filename):
    # This function takes the uploaded image and pass it to the inference function
    # It also saves the inferenced image and original image into their respective folders
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename.format(i)) for i in range(1, 2)]
    IMAGE_SIZE = (12, 8)

    stub = get_stub()       # Call the stub function containing the gRPC connection to TF serving

    for image_path in TEST_IMAGE_PATHS:
        img_org = Image.open(image_path)
        image_np = np.array(img_org)
        # image_np is passed onto the inference function which returns inferenced and 
        # original image and detected objects with their accuracy
        image_np_inferenced, image_np_original, detected_obj = inference(image_np, stub)
        # Inferenced image is converted from array to image and saved in detections folder
        im = Image.fromarray(image_np_inferenced)
        im.save('static/detection/' + filename)
        # original image is saved in the uploads folder
        img_org.save('static/uploads/' + filename)

    return detected_obj
   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)      # app will run on 0.0.0.0:5000