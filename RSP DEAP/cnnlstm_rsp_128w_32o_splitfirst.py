# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:59:27 2021

@author: Sowmya
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)

import time
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Permute
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking
#from keras.utils import plot_model

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Activation
from sklearn.model_selection import train_test_split


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import optimizers 
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

'''
# importing required modules
from zipfile import ZipFile
  
# specifying the zip file name
file_name = "lstm_slider128_rsp_raw_overlap32_withbaseline.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')

rsp_deap_data, rsp_deap_label = joblib.load(open('lstm_slider128_rsp_raw_overlap32_withbaseline.dat', 'rb'))
'''
rsp_data, rsp_label = joblib.load(open('Resp_32data_raw.dat', 'rb'))
rsp_data1 = np.array(rsp_data)   
rsp_label1 = np.array(rsp_label)  


def data_binarizer(ratings, threshold1, threshold2):
	"""binarizes the data below and above the threshold"""
	binarized = []
	for rating in ratings:
		if rating < threshold1:
			binarized.append(0)
		elif rating>= threshold2:
			binarized.append(1)
	return binarized

#convert binarized label (0 and 1) into categorical data- this generates 2 classes
from tensorflow.keras.utils import to_categorical
y_valence = np.array(data_binarizer([el[0] for el in rsp_label1],5,5))


#use stratify for split   
X_train_rsp_val, X_test_rsp_val, y_train_rsp_val, y_test_rsp_val = train_test_split(rsp_data1, y_valence, test_size=0.2, random_state=42, stratify=y_valence)
print('X_train_rsp_val', X_train_rsp_val.shape)
print('X_test_rsp_val', X_test_rsp_val.shape)
   


X_rsp_train_data = []
Y_rsp_train_label = []

for signal in range(0,X_train_rsp_val.shape[0]):
    #signal = signal[384:8064]
    #rsp_featurestrial = []
    #https://github.com/siddhi5386/Emotion-Recognition-from-brain-EEG-signals-/blob/master/Emotion_recognition_using_LSTM.ipynb
    start = 0;
    window_size = 128 #Averaging band power of 2 sec
    step_size = 32 #Each 0.125 sec update once
    while start + window_size < X_train_rsp_val.shape[1] - 32:
        X = X_train_rsp_val[signal][start : start + window_size]
        start = start + step_size 
        
        #rsp_featurestrial.append(X)
        Y_rsp_train_label.append(y_train_rsp_val[signal])
        X_rsp_train_data.append(X)  
   

X_rsp_train_data = np.array(X_rsp_train_data)   
Y_rsp_train_label = np.array(Y_rsp_train_label)  



X_rsp_test_data = []
Y_rsp_test_label = []

for signal in range(0,X_test_rsp_val.shape[0]):
    
    #https://github.com/siddhi5386/Emotion-Recognition-from-brain-EEG-signals-/blob/master/Emotion_recognition_using_LSTM.ipynb
    start = 0;
    window_size = 128 #Averaging band power of 2 sec
    step_size = 32 #Each 0.125 sec update once
    while start + window_size < X_test_rsp_val.shape[1] - 32:
        X = X_test_rsp_val[signal][start : start + window_size]
        start = start + step_size 
        
        #rsp_featurestrial.append(X)
        Y_rsp_test_label.append(y_test_rsp_val[signal])
        X_rsp_test_data.append(X)  
   

X_rsp_test_data = np.array(X_rsp_test_data)   
Y_rsp_test_label = np.array(Y_rsp_test_label)  


#Create balanced data
#oversample

import imblearn
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_rsp_train_resampled, y_rsp_train_resampled = ros.fit_resample(X_rsp_train_data, Y_rsp_train_label)
#X_ff_vald_resampled, y_ff_vald_resampled = ros.fit_resample(X_vald_ff_val, y_vald_ff_val)
X_rsp_test_resampled, y_rsp_test_resampled = ros.fit_resample(X_rsp_test_data, Y_rsp_test_label)
print("x_rsp_train_resampled",X_rsp_train_resampled.shape)
#print("x_ff_train_resampled",X_ff_vald_resampled.shape)
print("x_rsp_test_resampled",X_rsp_test_resampled.shape)

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_rsp_train_resampled)
print('Counter: y_rsp_train_resampled',counter) 
#counter = Counter(y_rsp_vald_resampled)
#print('Counter: y_rsp_vald_resampled',counter) 
counter = Counter(y_rsp_test_resampled)
print('Counter: y_rsp_test_resampled',counter) 

#convert binarized label (0, 1) into categorical data- this generates 2 classes
Z1 = np.ravel(y_rsp_train_resampled)
y_rsp_train_resampled = to_categorical(Z1)
#Z11 = np.ravel(y_ff_vald_resampled)
#y_ff_vald_resampled = to_categorical(Z11)
#y_test
Z22 = np.ravel(y_rsp_test_resampled)
y_rsp_test_resampled = to_categorical(Z22)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
rsp_training_set_scaled = sc.fit_transform(X_rsp_train_resampled)
#ff_vald_set_scaled = sc.transform(X_ff_vald_resampled)
rsp_testing_set_scaled = sc.transform(X_rsp_test_resampled)


n_features = 1
#order = F,  fortran like is important for separating 2 features (first 128 columns = 1 feature, next 128 columns -2nd feature), else every alternate column become 1 feature. 
x_rsp_train = rsp_training_set_scaled.reshape(rsp_training_set_scaled.shape[0], 128,n_features,  order='F' ) 
#x_ff_vald = ff_vald_set_scaled.reshape(ff_vald_set_scaled.shape[0], 128,n_features,  order='F' ) 
x_rsp_test = rsp_testing_set_scaled.reshape(rsp_testing_set_scaled.shape[0], 128, n_features, order='F')
print("x_rsp_train_reshaped", x_rsp_train.shape)
#print("x_rsp_vald_reshaped", x_rsp_vald.shape)
print("x_rsp_test_reshaped", x_rsp_test.shape)


GPUS = ["GPU:0", "GPU:1","GPU:2","GPU:3"] #https://github.com/jeffheaton/present/blob/master/youtube/gpu/keras-dual-gpu.ipynb
#strategy = tf.distribute.MirroredStrategy( GPUS)
strategy = tf.distribute.MirroredStrategy(devices = GPUS, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

start = time.time()
# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


#with tf.device('/GPU:0'):
with strategy.scope():
    model2 = Sequential()
    model2.add(Conv1D(filters=64, kernel_size=3,  strides=1, activation='relu', input_shape=(128,1)))
    model2.add(BatchNormalization())  
    #model2.add(Dropout(0.3))
    model2.add(Conv1D(filters=64, kernel_size=3,  strides=1, activation='relu'))
    model2.add(BatchNormalization())  
    #model2.add(Dropout(0.3))
    model2.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.3))
    model2.add(Bidirectional(LSTM(64, return_sequences=True)))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.3))
    model2.add(Bidirectional(LSTM(64)))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.3))
    model2.add(Dense(100, activation='relu'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.3))
    model2.add(Dense(2, activation='softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
mc = ModelCheckpoint('rawresp_splitfirst_500bs_8064_model2.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
model2.summary()
# fit network
history=model2.fit(x_rsp_train, y_rsp_train_resampled, epochs=200, batch_size=128, verbose=1, callbacks = [es,mc],validation_data=(x_rsp_test, y_rsp_test_resampled))
