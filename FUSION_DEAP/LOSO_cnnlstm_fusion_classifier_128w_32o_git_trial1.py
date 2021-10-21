# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 23:16:42 2021

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
from tensorflow.keras import optimizers 
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


GPUS = ["GPU:0", "GPU:1","GPU:2","GPU:3"] #https://github.com/jeffheaton/present/blob/master/youtube/gpu/keras-dual-gpu.ipynb
#strategy = tf.distribute.MirroredStrategy( GPUS)
strategy = tf.distribute.MirroredStrategy(devices = GPUS, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()) 
print('Number of devices: %d' % strategy.num_replicas_in_sync) 



# importing required modules
from zipfile import ZipFile
  
# specifying the zip file name

file_name1 = "lstm_slider128_emg1_raw_overlap32_withbaseline.zip"
# opening the zip file in READ mode
with ZipFile(file_name1, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
emg1_data, emg1_label = joblib.load(open('lstm_slider128_emg1_raw_overlap32_withbaseline.dat', 'rb'))


file_name2 = "lstm_slider128_emg2_raw_overlap32_withbaseline.zip"
# opening the zip file in READ mode
with ZipFile(file_name2, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
emg2_data, emg2_label = joblib.load(open('lstm_slider128_emg2_raw_overlap32_withbaseline.dat', 'rb'))

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
bvp_data, bvp_label = joblib.load(open('lstm_slider128_bvp_raw_overlap32_withbaseline.dat', 'rb'))

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
sub_ID, gsr_data, gsr_label = joblib.load(open('lstm_slider128_gsr_raw_overlap32__subID_withbaseline.dat', 'rb'))

file_name = "lstm_slider128_rsp_raw_overlap32_withbaseline.zip"
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
rsp_data, rsp_label = joblib.load(open('lstm_slider128_rsp_raw_overlap32_withbaseline.dat', 'rb'))

file_name = "lstm_slider128_eog1_raw_overlap32_withbaseline.zip"
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
eog1_data, eog1_label = joblib.load(open('lstm_slider128_eog1_raw_overlap32_withbaseline.dat', 'rb'))

file_name = "lstm_slider128_eog2_raw_overlap32_withbaseline.zip"
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
eog2_data, eog2_label = joblib.load(open('lstm_slider128_eog2_raw_overlap32_withbaseline.dat', 'rb'))

file_name = "lstm_slider128w_tmp_raw_32o_withbaseline.zip"
# opening the zip file in READ mode
with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
  
    # extracting all the files
    print('Extracting all the files now...')
    zip.extractall()
    print('Done!')
tmp_data, tmp_label = joblib.load(open('lstm_slider128w_tmp_raw_32o_withbaseline.dat', 'rb'))

'''
emg1_data, emg1_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/LSTM/EMG/lstm_slider128_emg1_raw_overlap32_withbaseline.dat', 'rb'))
emg2_data, emg2_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/LSTM/EMG/lstm_slider128_emg2_raw_overlap32_withbaseline.dat', 'rb'))
eog1_data, eog1_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/data/lstm_slider128_eog1_raw_overlap32_withbaseline.dat', 'rb'))
eog2_data, eog2_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/data/lstm_slider128_eog1_raw_overlap32_withbaseline.dat', 'rb'))
bvp_data, bvp_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/data/lstm_slider128_bvp_raw_overlap32_withbaseline.dat', 'rb'))
rsp_data, rsp_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/data/lstm_slider128_rsp_raw_overlap32_withbaseline.dat', 'rb'))
sub_ID, gsr_data, gsr_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/LSTM/GSR/lstm_slider128_gsr_raw_overlap32_subID_withbaseline.dat', 'rb'))
tmp_data, tmp_label = joblib.load(open('C:/Users/Sowmya/OneDrive - Athlone Institute Of Technology/PhD/DEAP/DATABASE/Python code/data/lstm_slider128w_tmp_raw_32o_withbaseline.dat', 'rb'))
'''

fusion_data = np.hstack((emg1_data ,emg2_data, eog1_data, eog2_data, bvp_data, rsp_data, tmp_data, gsr_data ))

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

from collections import Counter
 # summarize observations by class labeL
counter = Counter(y_valence)
print(counter)



fusion_data = pd.DataFrame(fusion_data)
raw_subid = pd.DataFrame(sub_ID)
fusion_data['y_val'] = pd.DataFrame(y_valence)
fusion_data['subID'] = raw_subid
print(fusion_data.shape)

accuracy = []
loss = []

for i in range(1,33):
    print("test subject : ", i)
    LOOCV_O = str(i)
    fusion_data['subID'] = fusion_data['subID'].apply(str)
    data_filtered = fusion_data[fusion_data['subID'] != LOOCV_O]
    data_test = fusion_data[fusion_data['subID'] == LOOCV_O]
    
    # Test data - the person left out of training
    data_test = data_test.drop(columns=['subID'])
    X_test = np.array(data_test.drop(columns=['y_val']))
    
    y_test = np.array(data_test['y_val']) #This is the outcome variable
    print(X_test.shape)   
    # Train data - all other people in dataframe
    data_train = data_filtered.drop(columns=['subID'])
    
    X_train = data_train.drop(columns=['y_val'])
    #feature_list = list(X_train.columns)
    
    X_train= np.array(X_train)
    y_train = np.array(data_train['y_val']) #Outcome variable here
    print(X_train.shape)
    print(y_train.shape)


    # Create balanced data
    #oversample
    import imblearn
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=0)
    X_ff_train_resampled, y_ff_train_resampled = ros.fit_resample(X_train, y_train)
    #X_ff_vald_resampled, y_ff_vald_resampled = ros.fit_resample(X_vald_ff_val, y_vald_ff_val)
    X_ff_test_resampled, y_ff_test_resampled = ros.fit_resample(X_test, y_test)
    print(X_ff_train_resampled.shape)
    #print(X_ff_vald_resampled.shape)
    print(X_ff_test_resampled.shape)
    
    from collections import Counter
     # summarize observations by class labeL
    counter = Counter(y_ff_train_resampled)
    print(counter) 
    #counter = Counter(y_ff_vald_resampled)
    #print(counter) 
    counter = Counter(y_ff_test_resampled)
    print(counter) 


    #convert binarized label (0, 1) into categorical data- this generates 2 classes
    Z1 = np.ravel(y_ff_train_resampled)
    y_ff_train_resampled = to_categorical(Z1)
    #Z11 = np.ravel(y_ff_vald_resampled)
    #y_ff_vald_resampled = to_categorical(Z11)
    #y_test
    Z22 = np.ravel(y_ff_test_resampled)
    y_ff_test_resampled = to_categorical(Z22)
    
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    ff_training_set_scaled = sc.fit_transform(X_ff_train_resampled)
    #ff_vald_set_scaled = sc.transform(X_ff_vald_resampled)
    ff_testing_set_scaled = sc.transform(X_ff_test_resampled)
    
    n_features = 8
    #order = F,  fortran like is important for separating 2 features (first 128 columns = 1 feature, next 128 columns -2nd feature), else every alternate column become 1 feature. 
    x_ff_train = ff_training_set_scaled.reshape(ff_training_set_scaled.shape[0], 128,n_features,  order='F' ) 
    #x_ff_vald = ff_vald_set_scaled.reshape(ff_vald_set_scaled.shape[0], 128,n_features,  order='F' ) 
    x_ff_test = ff_testing_set_scaled.reshape(ff_testing_set_scaled.shape[0], 128, n_features, order='F')
    print(x_ff_train.shape)
    #print(x_ff_vald.shape)
    print(x_ff_test.shape)


    start = time.time()
    # Nicely formatted time string
    def hms_string(sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)

    with strategy.scope():
        
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
    #es = EarlyStopping(monitor='accuracy', mode='max', verbose=1, patience=70)
    #mc = ModelCheckpoint('LOSO_ff8channels_1000bs_128w_32o_model2_git.h5', monitor='accuracy', mode='max', verbose=1, save_best_only=True)
    model2.summary()
    # fit network
    history=model2.fit(x_ff_train, y_ff_train_resampled, epochs=10, batch_size=500, verbose=1, validation_data=(x_ff_test, y_ff_test_resampled))
    model2.save('LOSO_testsubject_' + i +'_ff8chnl_cnnlstm_500bs_128w_32o_model2_git.h5')
    elapsed = time.time()-start
    print ('Training time: {elapsed in mins)}', hms_string(elapsed))

    test_loss, test_acc = model2.evaluate(x_ff_test, y_ff_test_resampled, verbose=1)
    accuracy.append(test_acc)
    loss.append(test_loss)
    
    print('Accuracy : '+ str(test_acc))
    print('loss : '+str(test_loss))
    print('... subject' + str(i) + ' processing complete.')

# Compute mean and std RMSE, MAPE
meanacc = np.mean(accuracy)
stdacc = np.std(accuracy)
meanloss = np.mean(loss)
stdloss = np.std(loss)
    
# Print RMSE, MAPE
print('Mean Accuracy:' + str(meanacc))
print('Std Accuracy:' + str(stdacc))
print('Mean loss:' + str(meanloss))
print('Std loss:' + str(stdloss))