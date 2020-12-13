# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:55:07 2020

@author: 10596
"""
import numpy as np
from tensorflow.keras.models import load_model

try:
    model = load_model('model.h5r------------>>>>>>>>>') # if you don't want to run it, change it to model.h5
except:
    # process the data
    f = open('names.txt')
    table3tr = []
    table3te = []
    for i in f:
        temp1 = chr(ord(i[0])+32) + i[1:-1] + (12-len(i))*chr(123)
        temp2 = i[1:-1] + (13-len(i))*chr(123)
        table2tr = []
        table2te = []
        for j in temp1:
            temc = [0 for i in range(27)]
            temc[ord(j)-97] = 1
            table2tr.append(temc)
        for j in temp2:
            temc = [0 for i in range(27)]
            temc[ord(j)-97] = 1
            table2te.append(temc)        
        table3tr.append(table2tr)
        table3te.append(table2te)
    x_train = np.array(table3tr[0:1700]).astype('float32')
    y_train = np.array(table3te[0:1700]).astype('float32')
    x_test = np.array(table3tr[1700:]).astype('float32')
    y_test = np.array(table3te[1700:]).astype('float32')
    
    # Building model and Training
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    import tensorflow.keras
    from tensorflow.python.keras.datasets import mnist
    from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
    
    model = Sequential()
    model.add(LSTM(27, input_shape=(11,27),return_sequences=True))
    model.add(Dense(27, activation='softmax'))   
    model.summary()
    model.compile(loss=tensorflow.keras.losses.mean_squared_error,
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=11, epochs=100, validation_data=(x_test, y_test))
    model.save('model.h5')


    import matplotlib.pyplot as plt
    # Plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



#def findKthLargest(nums, k):
#    index = 0
#    pivot = nums[index]
#    small=[i for i in (nums[:index]+nums[index+1:])if i < pivot]
#    large=[i for i in (nums[:index]+nums[index+1:])if i >= pivot]
#    if k-1 == len(large):
#        return pivot
#    elif k-1<len(large):
#        return findKthLargest(large,k)
#    if k-1 > len(large):
#            return findKthLargest(small,k-len(large)-1)
#
#
#model = load_model('model.h5')   
#c = input('Please input the first char of a name: \n')                   
#inpu = np.zeros([1,1,27])
#inpu[0,0,ord(c)-97] = 1
#names = []             
#while len(names) <= 20:
#    name = [c]
#    for n in range(10):               
#        infer = model.predict(inpu)
##        infer2 = infer[0,0]
#        infern = infer[0,-1].tolist()
#        nexchr = chr(infern.index(findKthLargest(infern,np.random.randint(1,5))) + 97) #randnames[k][m]
#        if (nexchr == '{'):
#            break
#        name.append(nexchr)
#        inpu3 = np.zeros([1,1,27])
#        inpu3[0,0,ord(nexchr)-97] = 1
#        inpu = np.hstack((inpu,inpu3))
#    name = ''.join(name)
#    if len(name) <= 2:
#        pass                
#    elif name in names:
#        pass
#    else:
#        names.append(name)                 
#     
#print(names)
