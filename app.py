import streamlit as st
from PIL import Image
import io
import os
import base64
import numpy as np
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
import cv2


def take_frame():
    st.title("Training Page")
    st.write("Start your face training")

    name = st.text_input("Employee Name")

    if st.button("Start Recording"):
        frames=[]
        # Open the webcam
        cap = cv2.VideoCapture(0)
        st.write("Recording started...")
        
        # Create a placeholder for the video stream
        video_placeholder = st.empty()

        count=0
        while count<200 :
            ret, frame = cap.read()
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            # Display the frame in Streamlit as a live video stream
            video_placeholder.image(frame_rgb)

            count+=1
        cap.release()
        st.write("Recording finished!")
        
        #The first frame is saved in storage, to be shown on "database" page
        image=Image.fromarray(frames[0])   # from numpy to PIL
        image.save(f"single_image/{name}.jpg")
        try:
            #Update the list of people to be shown on database page, also used to track image label
            with open("Listofpeople.pkl","rb") as file:
                Listofpeople=pickle.load(file)
            if name not in Listofpeople:
                Listofpeople.append(name)
            with open("Listofpeople.pkl","wb") as file:
                pickle.dump(Listofpeople,file)
        except:
            Listofpeople=[name]
            with open("Listofpeople.pkl","wb") as file:
                pickle.dump(Listofpeople,file)
        
        frames=np.array(frames)
        cap.release()
        return frames

def crop_and_resize(frames):
    cropped_frame=[]
    st.write("Cropping and resizing image....")
    no_of_frames=frames.shape[0]    
    print(no_of_frames," no. of frames")
    for i in range(no_of_frames):
        image=frames[i]
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            print(i)
            (x, y, w, h) = faces[0]
            cropped_image = image[y:y+h, x:x+w]  
            cropped_image=cv2.resize(cropped_image,(128,128)) 
            cropped_frame.append(cropped_image)
            
        else:
            print(i,"face not found")
    print("Done cropping")
    frames=np.array(cropped_frame)
    return frames    # return the croppe and resized frames

# To add the new person's frames and labels to the dataset as well as shuffle, saved as xtrain.npy and ytrain.npy
def update_training_data(frames):
    st.write("Training data is being updated......")
    # decide the label to be given to image
    with open("Listofpeople.pkl","rb") as file:
        Listofpeople=pickle.load(file)
    label_value=len(Listofpeople)
    label=np.zeros(frames.shape[0],dtype=int)
    label[:]=label_value-1
    
    # update xtrain and ytrain
    try:# in case of empty dataset
        # update ytrain
        ytrain=np.load('ytrain.npy')
        label=np.concatenate((ytrain,label),axis=0)
        np.random.seed(42)
        indices = np.arange(label.shape[0])
        np.random.shuffle(indices)
        label=label[indices]
        np.save('ytrain.npy',label) 
            
        #update xtrain
        xtrain = np.load('xtrain.npy')
        xtrain = np.concatenate((xtrain, frames), axis=0)
        np.random.seed(42)
        indices = np.arange(xtrain.shape[0])
        np.random.shuffle(indices)
        xtrain=xtrain[indices]
        np.save('xtrain.npy', xtrain)  
    except:
        np.save('ytrain.npy',label)
        np.save('xtrain.npy', frames) 
    st.write("Training data updation finished")

 
def one_hot_encode(ytrain):
    st.write("Doing encoding of labels")
    with open("Listofpeople.pkl","rb") as file:
        Listofpeople=pickle.load(file)
    no_of_points=ytrain.shape[0]
    no_of_classes=len(Listofpeople) 
    encoded_ytrain=[]
    temp=np.zeros(no_of_classes,dtype=int)
    for i in range(no_of_points):
        encoded_ytrain.append(np.copy(temp))
        encoded_ytrain[i][ytrain[i]]=1
    ytrain=np.array(encoded_ytrain)
    st.write("Encoding finished")
    return ytrain
    
def train_model():
    st.write("model training started...")
    xtrain=np.load('xtrain.npy')
    ytrain=np.load('ytrain.npy')
    ytrain=one_hot_encode(ytrain)  #one hot encoding
    
    model=Sequential()
    model.add(Conv2D(128, 5, activation='relu', input_shape=(128,128,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, 5, activation='relu', input_shape=(128,128,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, 5, activation='relu', input_shape=(128,128,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(64, 5, activation='relu', input_shape=(128,128,3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    
    with open("Listofpeople.pkl","rb") as file:
        Listofpeople=pickle.load(file)
    no_of_classes=len(Listofpeople)
    
    model.add(Dense(no_of_classes, activation='softmax'))

    from keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='accuracy', min_delta=0.0, patience=0, verbose=1, mode='auto', baseline=None, restore_best_weights=True)

    model.fit(xtrain,ytrain,epochs=50,batch_size=25, callbacks=[early_stopping])
    
    if st.button("Try more epochs"):
        n=st.text_input('Enter no. of epochs')
        model.fit(xtrain,ytrain,epochs=int(n),batch_size=25,callbacks=[early_stopping])
    model.save('model.h5')
    st.write("model training finished!")


def show_image():
    try:
        # Showing the list of people trained. Eg. {"ujjawal", "rickey", "shivam"}    
        with open("Listofpeople.pkl","rb") as file:
            Listofpeople=pickle.load(file)
        st.text("List of people: " + str(Listofpeople))
        
        #showing the length of dataset
        ytrain = np.load('ytrain.npy')
        xtrain = np.load('xtrain.npy')
        st.write("xtrain shape: ", xtrain.shape)
        st.write("ytrain shape: ", ytrain.shape)
        # Show all the images in the directory on the "database" page
        image_directory = "single_image"
        image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(".jpg")]
        for image_file in image_files:
            image = Image.open(image_file)
            image_name = os.path.splitext(os.path.basename(image_file))[0]
            st.write("Name: ",image_name)
            st.image(image,width=200)
   
    except:
        pass
    
def test_page():
    if st.button("Test your face"):
        try:
            model=load_model('model.h5')
            cap = cv2.VideoCapture(0)  
            ret, frame = cap.read()
            test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cap.release()
            cv2.destroyAllWindows()
            st.image(test_image, width=200)
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(test_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                test_image = test_image[y:y+h, x:x+w]
                test_image=cv2.resize(test_image, (128,128))
                test_image.shape=(1,128,128,3)
                prediction=model.predict(test_image)
                value=np.argmax(prediction)
                with open("Listofpeople.pkl","rb") as file:
                    Listofpeople=pickle.load(file)
                st.text("List of people: " + str(Listofpeople))
                st.write(Listofpeople[value])

            else:
                st.write("No face detected")
        except:
            st.write("No training dataset yet")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training", "Testing", "Database"], key="k1")

if page == "Training":
  #from tut3 import take_frame
  frames=take_frame()    #record and return the frames as numpy array eg. (400, 480, 640, 3)
  try:   # in case training button is not clicked in take_frame()
    #from tut2 import crop_and_resize
    frames=crop_and_resize(frames)    #return the face cropped and resized to (400,128,128,3)
    #from tut2 import update_training_data
    update_training_data(frames)      # new frames and their label added to xtrain and ytrain
    #from tut5 import train_model
    train_model()
  except:
    pass
     
if page=="Database":
  st.write("database is heree")
  #from tut4 import show_image
  show_image()
  
if page=="Testing":
  st.write("Testing is heree")
  #from tut6 import test_page
  test_page()
  
