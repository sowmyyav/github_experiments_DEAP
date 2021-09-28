# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:06:23 2021

@author: Sowmya
"""

import pandas as pd
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
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, BatchNormalization
#from keras.utils import plot_model

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Activation
from sklearn.model_selection import train_test_split


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

GPUS = ["GPU:0", "GPU:1","GPU:2","GPU:3"] #https://github.com/jeffheaton/present/blob/master/youtube/gpu/keras-dual-gpu.ipynb
#strategy = tf.distribute.MirroredStrategy( GPUS)
strategy = tf.distribute.MirroredStrategy(devices = GPUS, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
print('Number of devices: %d' % strategy.num_replicas_in_sync) 


# importing required modules
from zipfile import ZipFile
  
# specifying the zip file name
file_name = "lstm_slider128_gsr_raw_overlap32_withbaseline.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')


gsr_data, gsr_label = joblib.load(open('lstm_slider128_gsr_raw_overlap32_withbaseline.dat', 'rb'))

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

y_valence = np.array(data_binarizer([el[0] for el in gsr_label],5,5))
Z1 = np.ravel(y_valence)
y_val = to_categorical(Z1)
y_val

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_valence)
print(counter)

#use stratify for split   
X_train_gsr_val, X_test_gsr_val, y_train_gsr_val, y_test_gsr_val = train_test_split(gsr_data, y_valence, test_size=0.2, random_state=42, stratify=y_valence)
print(X_train_gsr_val.shape)
print(X_test_gsr_val.shape)


#use stratify for split   
#X_test1_gsr_val, X_vald_gsr_val, y_test1_gsr_val, y_vald_gsr_val = train_test_split(X_test_gsr_val, y_test_gsr_val, test_size=0.5, random_state=42, stratify=y_test_gsr_val)
#print(X_vald_gsr_val.shape)
#print(X_test1_gsr_val.shape)


# Create balanced data
#oversample
import imblearn
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_gsr_train_resampled, y_gsr_train_resampled = ros.fit_resample(X_train_gsr_val, y_train_gsr_val)
#X_gsr_vald_resampled, y_gsr_vald_resampled = ros.fit_resample(X_vald_gsr_val, y_vald_gsr_val)
X_gsr_test1_resampled, y_gsr_test1_resampled = ros.fit_resample(X_test_gsr_val, y_test_gsr_val)
print(X_gsr_train_resampled.shape)
#print(X_gsr_vald_resampled.shape)
print(X_gsr_test1_resampled.shape)

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_gsr_train_resampled)
print(counter) 
#counter = Counter(y_gsr_vald_resampled)
#print(counter) 
counter = Counter(y_gsr_test1_resampled)
print(counter) 


#convert binarized label (0, 1) into categorical data- this generates 2 classes
Z1 = np.ravel(y_gsr_train_resampled)
y_gsr_train_resampled = to_categorical(Z1)
#Z11 = np.ravel(y_gsr_vald_resampled)
#y_gsr_vald_resampled = to_categorical(Z11)
#y_test
Z22 = np.ravel(y_gsr_test1_resampled)
y_gsr_test1_resampled = to_categorical(Z22)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
gsr_training_set_scaled = sc.fit_transform(X_gsr_train_resampled)
#gsr_vald_set_scaled = sc.transform(X_gsr_vald_resampled)
gsr_testing1_set_scaled = sc.transform(X_gsr_test1_resampled)

n_features = 1
#order = F,  fortran like is important for separating 2 features (first 128 columns = 1 feature, next 128 columns -2nd feature), else every alternate column become 1 feature. 
x_gsr_train = gsr_training_set_scaled.reshape(gsr_training_set_scaled.shape[0], gsr_training_set_scaled.shape[1],n_features ) 
#x_gsr_vald = gsr_vald_set_scaled.reshape(gsr_vald_set_scaled.shape[0], 128,n_features ) 
x_gsr_test1 = gsr_testing1_set_scaled.reshape(gsr_testing1_set_scaled.shape[0], gsr_testing1_set_scaled.shape[1], n_features)
print(x_gsr_train.shape)
#print(x_gsr_vald.shape)
print(x_gsr_test1.shape)

start = time.time()
# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

with strategy.scope():
    

    model3 = Sequential()
    model3.add(Conv1D(filters=64, kernel_size=3,  strides=1, padding="causal", activation='relu', input_shape=(x_gsr_train.shape[1],x_gsr_train.shape[2])))
    #model1.add(BatchNormalization())  
    model3.add(Dropout(0.3))
    model3.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    #model1.add(BatchNormalization())  
    model3.add(Dropout(0.3))
    model3.add(Bidirectional(LSTM(64, return_sequences=True)))
    model3.add(BatchNormalization())  
    model3.add(Dropout(0.3))
    model3.add(Bidirectional(LSTM(64)))
    model3.add(BatchNormalization())  
    model3.add(Dropout(0.3))
    model3.add(Dense(100, activation='relu', input_dim = 1))
    model3.add(BatchNormalization())
    model3.add(Dropout(0.3))
    model3.add(Dense(100, activation='relu'))
    model3.add(BatchNormalization())
    model3.add(Dropout(0.3))
    model3.add(Dense(2, activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
mc = ModelCheckpoint('gsr_2000bs_128w_32o_model3.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
model3.summary()
# fit network
#history=model3.fit(x_gsr_train, y_gsr_train_resampled, epochs=50, batch_size=2000, verbose=1, callbacks = [es,mc],validation_data=(x_gsr_vald, y_gsr_vald_resampled))
history=model3.fit(x_gsr_train, y_gsr_train_resampled, epochs=50, batch_size=500, verbose=1, callbacks = [es,mc],validation_data=(x_gsr_test1, y_gsr_test1_resampled))

elapsed = time.time()-start
print ('Training time: {elapsed in mins)}', hms_string(elapsed))
