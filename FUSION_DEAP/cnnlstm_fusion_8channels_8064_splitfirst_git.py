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

rsp_data, rsp_label = joblib.load(open('Resp_32data_raw.dat', 'rb'))
rsp_data = np.array(rsp_data)   
rsp_label = np.array(rsp_label)  

emg1_data, emg1_label = joblib.load(open('EMG1_32data_raw.dat', 'rb'))
emg1_data = np.array(emg1_data)   
emg1_label = np.array(emg1_label)  

emg2_data, emg2_label = joblib.load(open('EMG2_32data_raw.dat', 'rb'))
emg2_data = np.array(emg2_data)   
emg2_label = np.array(emg2_label)  

eog1_data, eog1_label = joblib.load(open('EOG1_32data_raw.dat', 'rb'))
eog1_data = np.array(eog1_data)   
eog1_label = np.array(eog1_label)  

eog2_data, eog2_label = joblib.load(open('EOG2_32data_raw.dat', 'rb'))
eog2_data = np.array(eog2_data)   
eog2_label = np.array(eog2_label)  

gsr_data, gsr_label = joblib.load(open('GSR_32data_raw.dat', 'rb'))
gsr_data = np.array(gsr_data)   
gsr_label = np.array(gsr_label)  

bvp_data, bvp_label = joblib.load(open('Pleth_32data_raw.dat', 'rb'))
bvp_data = np.array(bvp_data)   
bvp_label = np.array(bvp_label)  

#tmp_data, tmp_label = joblib.load(open('C:/Users/Sowmya/Desktop/github_experiments/FUSION_DEAP/Tmp_32data_raw.dat', 'rb'))
tmp_data, tmp_label = joblib.load(open('Tmp_32data_raw.dat', 'rb'))
tmp_data = np.array(tmp_data)   
tmp_label = np.array(tmp_label)  

fusion_data = np.hstack((emg1_data, emg2_data, eog1_data, eog2_data, bvp_data, rsp_data, tmp_data, gsr_data ))


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
y_valence = np.array(data_binarizer([el[0] for el in emg1_label],5,5))


#use stratify for split   
X_train_ff_val, X_test_ff_val, y_train_ff_val, y_test_ff_val = train_test_split(fusion_data, y_valence, test_size=0.2, random_state=42, stratify=y_valence)
print('X_train_ff_val', X_train_ff_val.shape)
print('X_test_ff_val', X_test_ff_val.shape)

'''
#Create balanced data
#oversample
import imblearn
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_ff_train_resampled, y_ff_train_resampled = ros.fit_resample(X_train_ff_val, y_train_ff_val)
#X_ff_vald_resampled, y_ff_vald_resampled = ros.fit_resample(X_vald_ff_val, y_vald_ff_val)
X_ff_test_resampled, y_ff_test_resampled = ros.fit_resample(X_test_ff_val, y_test_ff_val)
print("x_ff_train_resampled",X_ff_train_resampled.shape)
#print("x_ff_train_resampled",X_ff_vald_resampled.shape)
print("x_ff_test_resampled",X_ff_test_resampled.shape)

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_ff_train_resampled)
print('Counter: y_ff_train_resampled',counter) 
#counter = Counter(y_ff_vald_resampled)
#print('Counter: y_ff_vald_resampled',counter) 
counter = Counter(y_ff_test_resampled)
print('Counter: y_ff_test_resampled',counter) 
'''
#convert binarized label (0, 1) into categorical data- this generates 2 classes
#Z1 = np.ravel(y_ff_train_resampled)
Z1 = np.ravel(y_train_ff_val)
y_ff_train_resampled = to_categorical(Z1)
#Z11 = np.ravel(y_ff_vald_resampled)
#y_ff_vald_resampled = to_categorical(Z11)
#y_test
#Z22 = np.ravel(y_ff_test_resampled)
Z22 = np.ravel(y_test_ff_val)
y_ff_test_resampled = to_categorical(Z22)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
ff_training_set_scaled = sc.fit_transform(X_train_ff_val)
#ff_vald_set_scaled = sc.transform(X_ff_vald_resampled)
ff_testing_set_scaled = sc.transform( X_test_ff_val)


n_features = 8
#order = F,  fortran like is important for separating 2 features (first 128 columns = 1 feature, next 128 columns -2nd feature), else every alternate column become 1 feature. 
x_ff_train = ff_training_set_scaled.reshape(ff_training_set_scaled.shape[0], 8064,n_features,  order='F' ) 
#x_ff_vald = ff_vald_set_scaled.reshape(ff_vald_set_scaled.shape[0], 128,n_features,  order='F' ) 
x_ff_test = ff_testing_set_scaled.reshape(ff_testing_set_scaled.shape[0], 8064, n_features, order='F')
print("x_ff_train_reshaped", x_ff_train.shape)
#print("x_ff_vald_reshaped", x_ff_vald.shape)
print("x_ff_test_reshaped", x_ff_test.shape)

start = time.time()
# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

GPUS = ["GPU:0", "GPU:1","GPU:2","GPU:3"] #https://github.com/jeffheaton/present/blob/master/youtube/gpu/keras-dual-gpu.ipynb
#strategy = tf.distribute.MirroredStrategy( GPUS)
strategy = tf.distribute.MirroredStrategy(devices = GPUS, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

with strategy.scope():
#with tf.device('/GPU:0'):
    
    #this was run with threshold 5 and added extra conv1d layer dropout after FIRST, second and third conv1d layer 
    
    model2 = Sequential()
    model2.add(Conv1D(filters=64, kernel_size=3,  strides=1, activation='relu', input_shape=(x_ff_train.shape[1],x_ff_train.shape[2])))
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
mc = ModelCheckpoint('ff8channels_500bs_8064_model2.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
model2.summary()
# fit network
history=model2.fit(x_ff_train, y_ff_train_resampled, epochs=300, batch_size=500, verbose=1, callbacks = [es,mc],validation_data=(x_ff_test, y_ff_test_resampled))

elapsed = time.time()-start
print ('Training time: {elapsed in mins)}', hms_string(elapsed))
