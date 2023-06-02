#!/usr/bin/env python
# coding: utf-8

# In[1]:


#GENERAL
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math


# In[2]:


#PATH PROCESS
import os
import os.path
from pathlib import Path
import glob
#IMAGE PROCESS
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
import imageio
from IPython.display import Image
import matplotlib.image as mpimg


# In[3]:


#MUSIC PROCESS
get_ipython().system('pip install pydub')
get_ipython().system('pip install librosa')
import pydub
from scipy.io.wavfile import read, write
import librosa
import librosa.display
import IPython
from IPython.display import Audio
import scipy
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder
#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# In[4]:


from keras.optimizers import RMSprop,Adam,Optimizer,Optimizer, SGD


# In[5]:


#MODEL LAYERS
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN,LSTM, GlobalAveragePooling2D, SeparableConv2D, ZeroPadding2D, Convolution2D, ZeroPadding2D,Reshape,Conv2DTranspose, LeakyReLU, Conv1D, AveragePooling1D, MaxPooling1D
from keras import models
from keras import layers
import tensorflow as tf
from keras.applications import VGG16,VGG19,inception_v3
from keras import backend as K
from keras.utils import plot_model
from keras.datasets import mnist
import keras


# In[6]:


#SKLEARN CLASSIFIER
get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install catboost')
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV


# In[7]:


#IGNORING WARNINGS
from warnings import filterwarnings
filterwarnings("ignore",category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning) 
filterwarnings("ignore", category=UserWarning)


# In[8]:


Main_Video_Path = Path("C:/Users/gunaa/Downloads/Real Life Violence Dataset")
Video_Path = list(Main_Video_Path.glob(r"*/*.mp4"))
Video_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],Video_Path))
Video_Path_Series = pd.Series(Video_Path,name="MP4").astype(str)
Video_Labels_Series = pd.Series(Video_Labels,name="CATEGORY")
Main_MP4_Data = pd.concat([Video_Path_Series,Video_Labels_Series],axis=1)


# In[9]:


Violence_Data = Main_MP4_Data[Main_MP4_Data["CATEGORY"] == "Violence"]
NonViolence_Data = Main_MP4_Data[Main_MP4_Data["CATEGORY"] == "NonViolence"]

Violence_Data = Violence_Data.reset_index()
NonViolence_Data = NonViolence_Data.reset_index()


# In[10]:


FPS = 30
DELAY = int(100/FPS)
# when it is necessary


# In[11]:


violence_frame_list = []

for file_video in Violence_Data.MP4:
    Video_File_Path = file_video
    
    Video_Caption = cv2.VideoCapture(Video_File_Path)
    Frame_Rate = Video_Caption.get(5)
    
    while Video_Caption.isOpened():
        
        Current_Frame_ID = Video_Caption.get(1)
        
        ret,frame = Video_Caption.read()
        
        if ret != True:
            break
            
        if Current_Frame_ID % math.floor(Frame_Rate) == 0:
            Frame_Resize = cv2.resize(frame,(64,64))
            violence_frame_list.append(Frame_Resize)
            
            
    Video_Caption.release()


# In[12]:


figure,axis = plt.subplots(5,5,figsize=(12,12))

for i,ax in enumerate(axis.flat):
    
    Img_Pick = violence_frame_list[i]
    
    ax.set_xlabel(Img_Pick.shape)
    ax.imshow(Img_Pick)

plt.tight_layout()
plt.show()


# In[13]:


X_4D_Violence = np.asarray(violence_frame_list)


# In[14]:


print(np.shape(X_4D_Violence))


# In[15]:


X_4D_Violence_Labels = np.ones((5832,1))


# In[16]:


print(np.shape(X_4D_Violence_Labels))


# In[17]:


X_4D_Violence_Labels = X_4D_Violence_Labels.flatten()


# In[18]:


X_4D_Violence_Labels = X_4D_Violence_Labels.astype(int)


# In[19]:


print(X_4D_Violence_Labels)


# In[20]:


nonviolence_frame_list = []

for file_video in NonViolence_Data.MP4:
    Video_File_Path = file_video
    
    Video_Caption = cv2.VideoCapture(Video_File_Path)
    Frame_Rate = Video_Caption.get(5)
    
    while Video_Caption.isOpened():
        
        Current_Frame_ID = Video_Caption.get(1)
        
        ret,frame = Video_Caption.read()
        
        if ret != True:
            break
            
        if Current_Frame_ID % math.floor(Frame_Rate) == 0:
            Frame_Resize = cv2.resize(frame,(64,64))
            nonviolence_frame_list.append(Frame_Resize)
            
            
    Video_Caption.release()


# In[21]:


figure,axis = plt.subplots(5,5,figsize=(12,12))

for i,ax in enumerate(axis.flat):
    
    Img_Pick = nonviolence_frame_list[i]
    
    ax.set_xlabel(Img_Pick.shape)
    ax.imshow(Img_Pick)

plt.tight_layout()
plt.show()


# In[22]:


X_4D_NonViolence = np.asarray(nonviolence_frame_list)


# In[23]:


print(np.shape(X_4D_NonViolence))


# In[24]:


X_4D_NonViolence_Labels = np.ones((4985,1))


# In[25]:


print(np.shape(X_4D_NonViolence_Labels))


# In[26]:


X_4D_NonViolence_Labels = X_4D_NonViolence_Labels.flatten()


# In[27]:


X_4D_NonViolence_Labels = X_4D_NonViolence_Labels.astype(int)


# In[28]:


print(np.shape(X_4D_NonViolence_Labels))


# In[29]:


print(X_4D_NonViolence_Labels)


# In[30]:


X_Train = np.concatenate((X_4D_Violence,X_4D_NonViolence),axis=0)


# In[31]:


print(np.shape(X_Train))


# In[32]:


Y_Train = np.concatenate((X_4D_Violence_Labels,X_4D_NonViolence_Labels),axis=0)


# In[33]:


print(np.shape(Y_Train))


# In[34]:


Target_X = X_Train
Label_X = Y_Train


# In[35]:


xTrain,xTest,yTrain,yTest = train_test_split(Target_X,Label_X,train_size=0.9,random_state=42,shuffle=True)


# In[36]:


print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)


# In[37]:


Model = Sequential()

Model.add(SeparableConv2D(12,(3,3),activation="relu",
                 input_shape=(64,64,3)))
Model.add(BatchNormalization())
Model.add(MaxPooling2D((2,2)))

#
Model.add(SeparableConv2D(24,(3,3),
                 activation="relu",padding="same"))
Model.add(Dropout(0.2))
Model.add(MaxPooling2D((2,2)))


#
Model.add(TimeDistributed(Flatten()))
Model.add(Bidirectional(LSTM(64,
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5)))
Model.add(Bidirectional(LSTM(64,
                                  return_sequences=True,
                                  dropout=0.5,
                                  recurrent_dropout=0.5)))

#
Model.add(Flatten())
Model.add(Dense(128,activation="relu"))
Model.add(Dropout(0.5))
Model.add(Dense(1,activation="sigmoid"))


# In[38]:


print(Model.summary())


# In[39]:


Callback_Stop_Early = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3)


# In[40]:


Model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


# In[41]:


CNN_Model = Model.fit(xTrain,yTrain,
                      validation_data=(xTest,yTest),
                      callbacks=Callback_Stop_Early,
                      epochs=50)

