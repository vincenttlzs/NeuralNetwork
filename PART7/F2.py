# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 20:53:08 2020

@author: 10596
"""


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
import tensorflow.keras
from tensorflow.python.keras.layers.core import Dense, Flatten

try:
    print(epochsssssss)# if you don't want to run it, delete it
except:   
    datas = np.loadtxt('dataset.txt')
    #plt.plot(datas[45500])
    
    y_train = datas[0:48000,0]
    x_train = datas[0:48000,1:101]
    
#    y_train = datas[:,0]
#    x_train = datas[:,1:101]
    
    y_test = datas[48001:50000,0]
    x_test = datas[48001:50000,1:101]
    
    y_train = tensorflow.keras.utils.to_categorical(y_train, 8)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 8)
    
    batch_size_x = 100
    epochs = 70
    
    model = Sequential()
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(150, activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.compile(loss=tensorflow.keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, batch_size=batch_size_x, epochs=epochs, validation_data=(x_test,y_test)) 

    # Plot
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.figure(2)
    plt.plot(history.history['accuracy'])
    plt.title('Accuracy')
    plt.ylabel('acuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')    
    plt.show()
    
testset = np.loadtxt('test.txt')   

pre = model.predict(testset)

f = open('FQ2Predicts_Zisheng_655677562.txt', 'w')
for i in range(len(pre)):
    temp = pre[i].tolist()
    f.write(str(temp.index(max(temp))))

f.close()
