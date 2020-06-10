#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
#for dirname, _, filenames in os.walk(r'/ml'):
 #   for filename in filenames:
  #      os.path.join(dirname, filename)
meta_data = pd.read_csv('../dataset/Meta.csv')
meta_shape = meta_data.shape
no_classes = meta_shape[0]
import cv2
train_data=[]
train_labels=[]

side = 20
channels = 3

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }
for c in range(no_classes) :
    path = "../dataset/train/{0}/".format(c)
    files = os.listdir(path)
    for file in files:
        train_image = cv2.imread(path+file)
        image_resized = cv2.resize(train_image, (side, side), interpolation = cv2.INTER_AREA)
        train_data.append(np.array(image_resized))
        train_labels.append(c)

data = np.array(train_data)
data = data.reshape((data.shape[0], 20*20*3))
data_scaled = data.astype(float)/255
labels = np.array(train_labels)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


labels = le.fit_transform(labels)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data_scaled, labels, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)


from sklearn import metrics
print(metrics.accuracy_score(y_pred, y_val))

from sklearn.externals import joblib
joblib.dump('traffic_classifier.h5','../dataset/traffic_classifier')

train_data1=[]

image = cv2.imread("../dataset/Train/6/00006_00000_00000.png")
image1 = cv2.resize(image, (side, side), interpolation = cv2.INTER_AREA)
data1=np.array(image1)
data1 = np.array(data1)
data1 = data1.reshape(1,( 20*20*3))

data_scaled1 = data1.astype(float)/255
k=data_scaled1
pred = model.predict(k)
print(pred[0])
sign = classes[pred[0]+1]
print(sign)


import tkinter as tk
from tkinter import filedialog
from tkinter import *

from PIL import ImageTk, Image

import numpy
#load the trained model to classify sign
#from keras.models import load_model
#model = joblib.load('../dataset/traffic_classifier')


#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)', 
            3:'Speed limit (50km/h)', 
            4:'Speed limit (60km/h)', 
            5:'Speed limit (70km/h)', 
            6:'Speed limit (80km/h)', 
            7:'End of speed limit (80km/h)', 
            8:'Speed limit (100km/h)', 
            9:'Speed limit (120km/h)', 
            10:'No passing', 
            11:'No passing veh over 3.5 tons', 
            12:'Right-of-way at intersection', 
            13:'Priority road', 
            14:'Yield', 
            15:'Stop', 
            16:'No vehicles', 
            17:'Veh > 3.5 tons prohibited', 
            18:'No entry', 
            19:'General caution', 
            20:'Dangerous curve left', 
            21:'Dangerous curve right', 
            22:'Double curve', 
            23:'Bumpy road', 
            24:'Slippery road', 
            25:'Road narrows on the right', 
            26:'Road work', 
            27:'Traffic signals', 
            28:'Pedestrians', 
            29:'Children crossing', 
            30:'Bicycles crossing', 
            31:'Beware of ice/snow',
            32:'Wild animals crossing', 
            33:'End speed + passing limits', 
            34:'Turn right ahead', 
            35:'Turn left ahead', 
            36:'Ahead only', 
            37:'Go straight or right', 
            38:'Go straight or left', 
            39:'Keep right', 
            40:'Keep left', 
            41:'Roundabout mandatory', 
            42:'End of no passing', 
            43:'End no passing veh > 3.5 tons' }

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = cv2.imread(file_path)
    image1 = cv2.resize(image, (side, side), interpolation = cv2.INTER_AREA)
    data1=np.array(image1)
    data1 = np.array(data1)
    data1 = data1.reshape(1,( 20*20*3))
    data_scaled1 = data1.astype(float)/255
    k=data_scaled1
    pred = model.predict(k)
    print(pred[0])
    sign = classes[pred[0]+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your Traffic Sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()