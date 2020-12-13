#!/usr/bin/env python

import numpy as np
import struct

# training images
train_images_idx3_ubyte_file = './image/train-images.idx3-ubyte'
# training labels
train_labels_idx1_ubyte_file = './image/train-labels.idx1-ubyte'

# testing images
test_images_idx3_ubyte_file = './image/t10k-images.idx3-ubyte'
# testing labels
test_labels_idx1_ubyte_file = './image/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    # binary data
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def run():
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    print 'load images and labels'
    w = learning(train_images,train_labels)
    errors = testing(w,test_images,test_labels)
    print errors
    print (float(errors)/10000)

def learning(img,lab,ny = 1,e = 13.8,n = 60000):
    w = np.random.rand(10,784)
    epoch = 0
    errors = []
    err = 0
    while(1): # epoch < 10000):
        indexs = []
        for i in range(n): # n the num of training image
            b = img[i].ravel() # unfold the pixels of a image
            vec = np.dot(w,b.reshape(-1,1)) #output of a image
            index  = 0
            max = vec[0]
            for j in range(len(vec)):
                if vec[j] >= max:
                    max = vec[j]
                    index = j
            indexs.append(index)
            if index != lab[i]:
                err += 1
        epoch += 1
        errors.append(err)
        print err
        for i in range(n):
            b = img[i].ravel() # unfold the pixels of a image
            vec = np.dot(w,b.reshape(-1,1)) #output of a image
            dx = np.zeros(10).reshape(-1, 1)
            dx[int(lab[i])] = 1
            uwx = np.zeros(10).reshape(-1, 1)
            for i in range(len(vec)):
                if vec[i] >= 0:
                    uwx[i] = 1
            w = w + ny*np.dot((dx - uwx),b.reshape(1,-1))
        # if(epoch%100 == 0):
        #     print ('epo: ',epoch,'err: ',err)
        #     np.savetxt("w_data.txt",w,fmt = "%f",delimiter=",")
        if (float(err)/float(n)) <= e:
            print('errors: ',errors)
            print('\nw: ',w)
            break
        err = 0
    np.savetxt("w_data.txt", w, fmt="%f", delimiter=",")
    return w

def testing(w,test_images,test_labels):
    errors = 0
    for i in range(len(test_images)):  # n the num of training image
        b = test_images[i].ravel()  # unfold the pixels of a image
        vec = np.dot(w, b.reshape(-1, 1))  # output of a image
        index = 0
        max = vec[0]
        for j in range(len(vec)):
            if vec[j] >= max:
                max = vec[j]
                index = j
        if index != test_labels[i]:
            errors += 1
    return errors



if __name__ == '__main__':
    run()
    print('end of pro')
