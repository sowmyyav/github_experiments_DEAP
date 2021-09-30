# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:25:13 2020

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
file_name = "lstm_slider128_bvp_raw_overlap32_withbaseline.zip"

# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')


bvp_data1, bvp_label1 = joblib.load(open('lstm_slider128_bvp_raw_overlap32_withbaseline.dat', 'rb'))


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

y_valence = np.array(data_binarizer([el[0] for el in bvp_label1],4.5,4.5))
Z1 = np.ravel(y_valence)
y_val = to_categorical(Z1)
y_val

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_valence)
print(counter)

#use stratify for split   

X_train_bvp_val, X_test_bvp_val, y_train_bvp_val, y_test_bvp_val = train_test_split(bvp_data1, y_valence, test_size=0.1, random_state=42, stratify=y_valence)
print(X_train_bvp_val.shape)
print(X_test_bvp_val.shape)

# Create balanced data
#oversample

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_bvp_train_resampled, y_bvp_train_resampled = ros.fit_resample(X_train_bvp_val, y_train_bvp_val)
X_bvp_test_resampled, y_bvp_test_resampled = ros.fit_resample(X_test_bvp_val, y_test_bvp_val)
print(X_bvp_train_resampled.shape)
print(X_bvp_test_resampled.shape)

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_bvp_train_resampled)
print(counter) 
counter = Counter(y_bvp_test_resampled)
print(counter) 


#convert binarized label (0, 1, 2) into categorical data- this generates 3 classes

Z1 = np.ravel(y_bvp_train_resampled)
y_bvp_train_resampled = to_categorical(Z1)
#y_test
Z2 = np.ravel(y_bvp_test_resampled)
y_bvp_test_resampled = to_categorical(Z2)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
bvp_training_set_scaled = sc.fit_transform(X_bvp_train_resampled)
bvp_testing_set_scaled = sc.transform(X_bvp_test_resampled)

n_features = 1
x_bvp_train = bvp_training_set_scaled.reshape(bvp_training_set_scaled.shape[0], bvp_training_set_scaled.shape[1],n_features ) 
x_bvp_test = bvp_testing_set_scaled.reshape(bvp_testing_set_scaled.shape[0], bvp_training_set_scaled.shape[1], n_features)
print(x_bvp_train.shape)
print(x_bvp_test.shape)

start = time.time()
# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

model2 = Sequential()
model2.add(Conv1D(filters=64, kernel_size=3,  strides=1, padding="causal", activation='relu', input_shape=(x_bvp_train.shape[1],x_bvp_train.shape[2])))
model2.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model2.add(Bidirectional(LSTM(64, return_sequences=True)))
model2.add(Dropout(0.3))
model2.add(Bidirectional(LSTM(128)))
model2.add(Dropout(0.3))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten())
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=50)
mc = ModelCheckpoint('BVP_256bs_128w_32o_model2.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
model2.summary()
# fit network
history=model2.fit(x_bvp_train, y_bvp_train_resampled, epochs=200, batch_size=256, verbose=1,validation_data=(x_bvp_test, y_bvp_test_resampled))

elapsed = time.time()-start
print ('Training time: {elapsed in mins)}', hms_string(elapsed))