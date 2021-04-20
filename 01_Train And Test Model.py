import pandas as pd
import numpy as np
from scipy.fft import fft, ifft
from scipy.stats import spearmanr as corr
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from pyyawt import waverec,wavedec,dwt,idwt,wnoise
from pyyawt.Denoising import wden

data1 = pd.read_excel("New_Poseidon_1.xlsx")

dataset = pd.DataFrame(data1)

dataset = dataset.drop(['DEPTH','ATRX','ATRT','HDAR','DTSM','DTCO','CALI'],axis=1) #remove column
dataset = dataset.drop(np.arange(0,3409,1)) #remove rows with no specified target
dataset = dataset.drop(np.arange(9060,9073,1)) #remove rows with no specified target
dataset = dataset[dataset.ECGR != -999.25]
dataset = dataset[dataset.NPHI != -999.25]
dataset = dataset[dataset.HROM != -999.25]
dataset = dataset.iloc[:,0:].values
dataset = pd.DataFrame(dataset)

# # # # # # # # # # SIMPLIFICATION FACIES # # # # # # # # # #

dataset.iloc[:,-1]= dataset.iloc[:,-1].replace('Argillaceous Siltstone', 'Siltstone')
dataset.iloc[:,-1]= dataset.iloc[:,-1].replace('Silty Claystone', 'Claystone')
dataset.iloc[:,-1]= dataset.iloc[:,-1].replace('Silty Sandstone', 'Sandstone')

#  Remaining Facies and its Frequency after Simplification
Facies_Freq = np.unique(dataset.iloc[:,-1], return_counts=True)
# print(Facies_Freq)

# # INPUT DATASET & TARGET
log = dataset.iloc[:,:-1].values
label = dataset.iloc[:,-1].astype('category')
label = label.cat.codes
# print(np.shape(log))
# print(label)

# # # # # # # # # # # # FILTERING LOGS # # # # # # # # # # #
# Using Discrete Wavelet Transform Denoising (Modul --> pyyawt by Holger Nahrstaedt)

numlog = np.shape(log[1])
level = 5

GR = log[:,0]
NPHI = log[:,1]
RHOB = log[:,2]

GR = np.array(GR, dtype=float)
NPHI = np.array(NPHI, dtype=float)
RHOB = np.array(RHOB, dtype=float)

# # WAVELET DENOISING FILTER # #

# GR #
[CLog, LLog] = wavedec(GR,level,'sym8')
[XDLog,CXDLog,LDXLog] = wden(CLog,LLog,'heursure','s','one',level,'sym8')
deno_log_GR = waverec(CXDLog,LDXLog,'sym8') # Use this for GR Input
# print(np.shape(deno_log_GR))

spec_log = fft(GR)
spec_log = abs(spec_log)

spec_deno_log_GR = fft(deno_log_GR)
spec_deno_log_GR = abs(spec_deno_log_GR)

# VISUALIZING BEFORE AND AFTER DENOISING

# f, ax = plt.subplots(nrows=4,ncols=2, figsize=(20,10))
# ax[0,0].plot(spec_log)
# ax[0,0].grid()
# ax[0,0].set_xlabel("GR Spectrum Before Denoising")
# ax[0,1].plot(spec_deno_log_GR)
# ax[0,1].grid()
# ax[0,1].set_xlabel("GR Spectrum After Denoising")

# NPHI #
[CLog1, LLog1] = wavedec(NPHI,level,'sym8')
[XDLog1,CXDLog1,LDXLog1] = wden(CLog1,LLog1,'heursure','s','one',level,'sym8')
deno_log_NPHI = waverec(CXDLog1,LDXLog1,'sym8') # Use this for NPHI input

spec_log1 = fft(NPHI)
spec_log1 = abs(spec_log1)

spec_deno_log_NPHI = fft(deno_log_NPHI)
spec_deno_log_NPHI = abs(spec_deno_log_NPHI)

# VISUALIZING BEFORE AND AFTER DENOISING

# ax[1,0].plot(spec_log1)
# ax[1,0].grid()
# ax[1,0].set_xlabel("NPHI Spectrum Before Denoising")
# ax[1,1].plot(spec_deno_log_NPHI)
# ax[1,1].grid()
# ax[1,1].set_xlabel("NPHI Spectrum After Denoising")

# RHOB #
[CLog2, LLog2] = wavedec(RHOB,level,'sym8')
[XDLog2,CXDLog2,LDXLog2] = wden(CLog2,LLog2,'heursure','s','one',level,'sym8')
deno_log_RHOB = waverec(CXDLog2,LDXLog2,'sym8') # Use this for RHOB input

spec_log2 = fft(RHOB)
spec_log2 = abs(spec_log2)

spec_deno_log_RHOB = fft(deno_log_RHOB)
spec_deno_log_RHOB = abs(spec_deno_log_RHOB)

# VISUALIZING BEFORE AND AFTER DENOISING

# ax[2,0].plot(spec_log2)
# ax[2,0].grid()
# ax[2,0].set_xlabel("RHOB Spectrum Before Denoising")
# ax[2,1].plot(spec_deno_log_RHOB)
# ax[2,1].grid()
# ax[2,1].set_xlabel("RHOB Spectrum After Denoising")

# NOISE PROFILE
noise_GR = spec_log - spec_deno_log_GR
noise_NPHI = spec_log1 - spec_deno_log_NPHI
noise_RHOB = spec_log2 - spec_deno_log_RHOB

# VISUALIZING NOISE

# f, ax = plt.subplots(nrows=4,ncols=1, figsize=(20,10))
# ax[0].plot(noise_GR)
# ax[0].grid()
# ax[0].set_ylabel("Log GR")
# ax[1].plot(noise_NPHI)
# ax[1].grid()
# ax[1].set_ylabel("Log NPHI")
# ax[2].plot(noise_RHOB)
# ax[2].grid()
# ax[2].set_ylabel("Log RHOB")
# ax[3].plot(noise_CALI)
# ax[3].grid()
# ax[3].set_ylabel("Log CALI")
# f.suptitle('SPECTRUM NOISE', fontsize=14, y=0.94)
# plt.show()

# # # # # # # # # # # # CONDITIONING GR LOG # # # # # # # # #

l = len(deno_log_GR)
m = 50               # cut-off 0-50 API --> Sandstone, Limestone, Dolomite, Calcilutite, Calcarenite
n = 150              # cut-off 50-150 API --> Shale (Siltstone, Claystone), 150-  API --> Organic-Rich Shale

for i in range(l):
    if deno_log_GR[i] <= m:
        deno_log_GR[i] = 1
    elif m < deno_log_GR[i] <= n:
        deno_log_GR[i] = 2
    else:
        deno_log_GR[i] = 3

deno_log_GR = np.array(deno_log_GR,dtype=int)

# # # # # # # CREATE ARTIFICIAL LOGS FROM EXISTING LOGS # # # # # # # #
# Performing Mathematical Operation (Logarithmic, 1/Logarithmic, Square root) toward all Logs

# LOGS FROM NPHI
logh_NPHI = np.log(deno_log_NPHI)
ilogh_NPHI = 1/logh_NPHI
sq_NPHI = np.sqrt(deno_log_NPHI)

# LOGS FROM RHOB
logh_RHOB = np.log(deno_log_RHOB)
ilogh_RHOB = 1/logh_RHOB
sq_RHOB = np.sqrt(deno_log_RHOB)


# # DATASET FINAL # #
dataset_final = pd.DataFrame(list(zip(deno_log_GR,deno_log_NPHI,logh_NPHI,ilogh_NPHI,sq_NPHI,
                                      deno_log_RHOB,logh_RHOB,ilogh_RHOB,sq_RHOB)))

dataset_final = dataset_final.values

# # # # # # # CNN MODEL BUILDING & TRAINING DATA # # # # # #

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
dataset_final = sc.fit_Transform(dataset_final)
dataset_final = np.expand_dims(dataset_final, axis=2)

# TARGET ENCODING --> ONE-HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
target = dataset.iloc[:,-1].values
target = np.reshape(target,(1200,1))
target_ = onehotencoder.fit_Transform(target).toarray()
# print(np.shape(target))

# SHUFFLE THE DATASET
from sklearn.utils import shuffle
dataset_final,target_ = shuffle(dataset_final,target_)

# # # # # # # BUILD AND TRAIN CNN MODEL # # # # # # #

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import time

# Hyperparameter
dense_layers = [1]
layer_sizes = [256]
conv_layers = [1]

for dense_layer in dense_layers:
        for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                        NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,
                                                                     int(time.time()))
                        tensorboard = TensorBoard(log_dir='logs_with_CALI_13/{}'.format(NAME))
                        # print(NAME)

                        # Initialising the CNN
                        cnn = tf.keras.models.Sequential()

                        # # Step 1 - Convolution
                        for i in range(conv_layer):
                                cnn.add(tf.keras.layers.Conv1D(filters=layer_size, kernel_size=2,
                                                               input_shape=[9, 1], padding='same'))
                                cnn.add(tf.keras.layers.LeakyReLU(alpha=0.1))

                        # # Step 2 - Pooling
                        cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=1))

                        # Step 3 - Flattening
                        cnn.add(tf.keras.layers.Flatten())

                        # Step 4 - Full Connection
                        for i in range (dense_layer):
                                cnn.add(tf.keras.layers.Dense(units=layer_size))
                                cnn.add(tf.keras.layers.LeakyReLU(alpha=0.1))

                        #Step 5 - DROPOUT Layers
                        cnn.add(tf.keras.layers.Dropout(0.5))
                        cnn.add(tf.keras.layers.Dense(units=128))
                        cnn.add(tf.keras.layers.LeakyReLU(alpha=0.1))

                        # Step 5 - Output Layer
                        cnn.add(tf.keras.layers.Dense(units=5, activation='softmax'))

                        # Part 3 - Training the CNN

                        # Compiling the CNN
                        cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                                    metrics = ['accuracy'])

                        # Training the CNN on the Training set and evaluating it on the Test set
                        cnn.fit(x = dataset_final,y = target_, validation_split = 0.2, epochs = 50,
                                batch_size = 10, verbose =1) #, callbacks=[tensorboard]
                        cnn.save('Poseidon_1_Thin_Train_5_9.model')
                        cnn.summary()
                        break

# # # # # # # # # # # # CNN PREDICTION (TEST MODEL) # # # # # # # # # # # #

# LOAD MODEL
cnn2 = tf.keras.models.load_model("Poseidon_1_Thin_Train_5_9.model") # Changeable

from sklearn.preprocessing import LabelEncoder

Encode = LabelEncoder()
Encode.fit(target)

# PREPARE DATA FOR PREDICTION

Well_Poseidon2 = pd.read_excel('New_Poseidon_2.xlsx')
Well_Poseidon2 = Well_Poseidon2.drop(['DEPTH','RS','RD','DTSM','DTCO','DCAV'],axis=1)
# print(Well_Poseidon2)

# REMOVE ROWS CONTAINING -999.25 (-999.25 MEANS NO DATA IN CERTAIN RANGE)
Well_Poseidon2 = Well_Poseidon2.drop(np.arange(9615,9722,1)) #remove rows with no specified target
Well_Poseidon2 = Well_Poseidon2[Well_Poseidon2.GR != -999.25]
Well_Poseidon2 = Well_Poseidon2[Well_Poseidon2.TNP != -999.25]
Well_Poseidon2 = Well_Poseidon2[Well_Poseidon2.RHOB != -999.25]

Dataset_Pos2 = Well_Poseidon2.iloc[:, [0,1,2]].values #Contain GR, TNP, RHOB

GR_Target = Dataset_Pos2[:,0]
NPHI_Target = Dataset_Pos2[:,1]
RHOB_Target = Dataset_Pos2[:,2]

GR_Target = np.array(GR_Target, dtype=float)
NPHI_Target = np.array(NPHI_Target, dtype=float)
RHOB_Target = np.array(RHOB_Target, dtype=float)

# # # # # # # # WAVELET DENOISING FILTER # # # # # # #

# GR #
[CLog3, LLog3] = wavedec(GR_Target,level,'sym8')
[XDLog3,CXDLog3,LDXLog3] = wden(CLog3,LLog3,'heursure','s','one',level,'sym8')
deno_log_GR_Target = waverec(CXDLog3,LDXLog3,'sym8')

# NPHI #
[CLog4, LLog4] = wavedec(NPHI_Target,level,'sym8')
[XDLog4,CXDLog4,LDXLog4] = wden(CLog4,LLog4,'heursure','s','one',level,'sym8')
deno_log_NPHI_Target = waverec(CXDLog4,LDXLog4,'sym8') # Use this for NPHI input

# RHOB #
[CLog5, LLog5] = wavedec(RHOB_Target,level,'sym8')
[XDLog5,CXDLog5,LDXLog5] = wden(CLog5,LLog5,'heursure','s','one',level,'sym8')
deno_log_RHOB_Target = waverec(CXDLog5,LDXLog5,'sym8') # Use this for RHOB input

# # CONDITIONING GR LOG

# LOGS FROM GR
l2 = len(deno_log_GR_Target)

for i in range(l2):
    if deno_log_GR_Target[i] <= m:
        deno_log_GR_Target[i] = 1
    elif m < deno_log_GR_Target[i] <= n:
        deno_log_GR_Target[i] = 2
    else:
        deno_log_GR_Target[i] = 3

# # # # # # # # # CREATE ARTIFICIAL LOG FROM EXISTING LOG # # # # # # # #

# LOGS FROM NPHI
logh_NPHI_Target = np.log(deno_log_NPHI_Target)
ilogh_NPHI_Target = 1/logh_NPHI_Target
sq_NPHI_Target = np.sqrt(deno_log_NPHI_Target)

# LOGS FROM RHOB
logh_RHOB_Target = np.log(deno_log_RHOB_Target)
ilogh_RHOB_Target = 1/logh_RHOB_Target
sq_RHOB_Target = np.sqrt(deno_log_RHOB_Target)

# # DATASET FINAL
dataset_final_target = pd.DataFrame(list(zip(deno_log_GR_Target,deno_log_NPHI_Target,logh_NPHI_Target,ilogh_NPHI_Target,sq_NPHI_Target
                                             ,deno_log_RHOB_Target,logh_RHOB_Target,ilogh_RHOB_Target,sq_RHOB_Target)))
dataset_final_target = dataset_final_target.values

# FEATURE SCALING
dataset_final_target = sc.fit_Transform(dataset_final_target)
dataset_final_target = np.expand_dims(dataset_final_target, axis=2)

# print(np.shape(dataset_final_target))

prediction = cnn2.predict_classes(dataset_final_target).reshape(2423,1)
# # print(prediction)
# # print(np.shape(prediction))
prediction_ = Encode.inverse_Transform(prediction).reshape(2423,1)

Facies_pred = pd.DataFrame(prediction_)

# # # # # # STORE THE PREDICTION RESULT TO EXCEL # # # # # #

with pd.ExcelWriter('New_Poseidon_2.xlsx',
                    mode='a') as writer:
     Facies_pred.to_excel(writer, sheet_name='Thin-Train-9-5-10')

# # # # NOTE : PERFORM THIS WHOLE SCRIPT MULTIPLE TIMES TO GET MULTIPLE RESULT # # # #