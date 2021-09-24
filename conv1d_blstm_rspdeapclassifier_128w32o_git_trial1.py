# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:34:32 2021

@author: Sowmya
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, KFold, RepeatedStratifiedKFold
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Permute
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking
#from keras.utils import plot_model

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Activation
from sklearn.model_selection import train_test_split


from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import optimizers 
from numpy.random import seed
seed(0)
import tensorflow as tf
tf.random.set_seed(0)

import time


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
#rsp_deap_data, rsp_deap_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/data/lstm_slider128_rsp_raw_overlap32_withbaseline.dat', 'rb'))
#convert raw label into categorical data- this generates 10 classes
from tensorflow.keras.utils import to_categorical

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
y_valence = np.array(data_binarizer([el[0] for el in rsp_deap_label],5,5))
Z1 = np.ravel(y_valence)
y_train1 = to_categorical(Z1)
y_train1


from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_valence)
print(counter)

# Create balanced data

#use stratify for split   
X_train_rsp_val, X_test_rsp_val, y_train_rsp_val, y_test_rsp_val = train_test_split(rsp_deap_data, y_valence, test_size=0.2, random_state=42, stratify=y_valence)

#oversample

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_rsp_val, y_train_rsp_val)
X_test_resampled, y_test_resampled = ros.fit_resample(X_test_rsp_val, y_test_rsp_val)


from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_train_resampled)
print(counter) 

#convert binarized label (0 and 1) into categorical data- this generates 2 classes
#y_valence = np.array(data_binarizer([el[0] for el in rsp_deap_label],5,5))
Z1 = np.ravel(y_train_resampled)
y_train_resampled = to_categorical(Z1)
#y_train1
Z2 = np.ravel(y_test_resampled)
y_test_resampled = to_categorical(Z2)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_set_scaled = sc.fit_transform(X_train_resampled)
testing_set_scaled = sc.transform(X_test_resampled)

#sc.data_min_
#sc.data_max_
n_features = 1
x_train = training_set_scaled.reshape(training_set_scaled.shape[0], training_set_scaled.shape[1],n_features )
x_test = testing_set_scaled.reshape(testing_set_scaled.shape[0], testing_set_scaled.shape[1], n_features)
print(x_train.shape)
print(x_test.shape)

#Best model - 65% accuracy on balanced resp deap data
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

model1 = Sequential()
model1.add(Conv1D(filters=64, kernel_size=3,  strides=1, padding="causal", activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
#model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model1.add(Bidirectional(LSTM(64, return_sequences=True)))
model1.add(Dropout(0.3))
model1.add(Bidirectional(LSTM(64)))
model1.add(Dropout(0.3))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten())
model1.add(Dense(100, activation='relu'))
model1.add(Dropout(0.3))
model1.add(Dense(2, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70)
mc = ModelCheckpoint('rspdeap_256bs_128w_32o_model1_epoch500_git.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
# fit network
history=model1.fit(x_train, y_train_resampled, epochs=500, batch_size=256, verbose=2,validation_data=(x_test, y_test_resampled))
