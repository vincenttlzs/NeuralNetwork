import numpy as np
import math
import random
import matplotlib.pyplot as plt
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

mid = 150
class neuronNetwork():

    def __init__(self,neuNumList= [[mid,784], [mid,1], [mid,10], [10,1]],ny = 0.1):
        self.neuNumList = neuNumList
        self.neuronInit()
        self.ny = ny

        self.dict = {'0.0': np.array([1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                     '1.0': np.array([-1.0, 1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                     '2.0': np.array([-1.0, -1.0, 1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                     '3.0': np.array([-1.0, -1.0, -1.0, 1.0,-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                     '4.0': np.array([-1.0, -1.0, -1.0, -1.0, 1.0,-1.0, -1.0, -1.0, -1.0, -1.0]),
                     '5.0': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, 1.0,-1.0, -1.0, -1.0, -1.0]),
                     '6.0': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,-1.0, -1.0, -1.0]),
                     '7.0': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,-1.0, -1.0]),
                     '8.0': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,-1.0]),
                     '9.0': np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0]),
                     }

    def neuronInit(self):
        w = []
        for i in range(len(self.neuNumList)):
            wi = np.random.rand(self.neuNumList[i][0],self.neuNumList[i][1])*2 -1
            w.append(wi)
        self.w = w

    def training(self,xn,dn):
        self.xn = xn
        self.dn = dn
        num = len(self.xn)
        errs = []
        for e in range(300):
            err = 0
            for n in range(num):
                temp =  self.w[1] + np.dot(self.w[0],self.xn[n].reshape(-1, 1))
                # hiddening layer
                v1 = []
                for j in range(temp.shape[0]):
                    y = self.acfun(temp[j,0])
                    v1.append(y)
                # output later
                v1 = np.array(v1,ndmin=2)
                temp2 = np.dot(self.w[2].T,v1.T) + self.w[3]
                v2 = []
                for j in range(temp2.shape[0]):
                    y = self.acfun(temp2[j,0])
                    v2.append(y)
                index = 0
                max = v2[0]
                for j in range(len(v2)):
                    if v2[j] >= max:
                        max = v2[j]
                        index = j
                if index != self.dn[n]:
                    err += 1
            print err
            errs.append(err)

            if (float(err) / num) <=0.05:
                break

            if self.ny >=0.0015:
                self.ny = self.ny*0.95

            for n in range(num):
                self.updateW(n)

        self.errs = errs

    def updateW(self,n):
        # hidening layer
        temp =  np.dot(self.w[0],self.xn[n].reshape(-1, 1)) + self.w[1]
        v1 = []
        tanpv1 = []
        for j in range(temp.shape[0]):
            y = self.acfun(temp[j,0])
            z = self.tanhp(temp[j,0])
            v1.append(y)
            tanpv1.append(z)
        v1 = np.array(v1,ndmin=2)
        tanpv1 = np.array(tanpv1)
        # output layer
        temp2 = np.dot(self.w[2].T,v1.T) + self.w[3]
        v2 = []
        tanpv2 = []
        for j in range(temp2.shape[0]):
            y = self.acfun(temp2[j,0])
            z = self.tanhp(temp2[j,0])
            v2.append(y)
            tanpv2.append(z)
        v2 = np.array(v2)
        tanpv2 = np.array(tanpv2)
        dd = self.dict[str(self.dn[n])] - v2
        dd = np.array(dd)
        # update w1 matrix
        tw = np.dot(np.array(tanpv1*(np.dot(self.w[2], tanpv2*dd)),ndmin=2).T,np.array(self.xn[n],ndmin=2))
        # # update 1th bias matrix
        tb = np.array(tanpv1*(np.dot(self.w[2], tanpv2*dd)),ndmin=2).T
        # update w2 matrix
        tu = np.dot(np.array(v1,ndmin=2).T,np.array(tanpv2*dd,ndmin=2))

        # # update 2th bias matrix
        tb2 = np.array(tanpv2*dd,ndmin=2).T

        self.w[0] = self.w[0] + 2*self.ny*tw
        self.w[1] = self.w[1] + 2*self.ny*tb
        self.w[2] = self.w[2] + 2*self.ny*tu
        self.w[3] = self.w[3] + 2*self.ny * tb2


    def acfun(self,m):
        return math.tanh(m)

    def tanhp(self,v):
        return 1 - pow(self.acfun(v),2)/(v*v)

    def testing(self,xn,dn):
        print ( 'testing ')
        self.xn = xn
        self.dn = dn
        num = len(self.xn)
        err = 0
        for n in range(num):
            temp = self.w[1] + np.dot(self.w[0], self.xn[n].reshape(-1, 1))
            # hiddening layer
            v1 = []
            for j in range(temp.shape[0]):
                y = self.acfun(temp[j, 0])
                v1.append(y)
            # output later
            v1 = np.array(v1, ndmin=2)
            temp2 = np.dot(self.w[2].T, v1.T) + self.w[3]
            v2 = []
            for j in range(temp2.shape[0]):
                y = self.acfun(temp2[j, 0])
                v2.append(y)
            # v2 = np.dot(v1,self.w[2]) + self.w[3]
            index = 0
            max = v2[0]
            for j in range(len(v2)):
                if v2[j] >= max:
                    max = v2[j]
                    index = j
            if index != self.dn[n]:
                err += 1
        return err


    def plotCurve(self):
        x = np.linspace(0, 1, 100)
        output = []
        for n in range(len(x)):
            temp = self.w[1] + np.dot(x[n], self.w[0])
            vi = []
            for j in range(temp.shape[0]):
                y = self.acfun(temp[j])
                vi.append(y)
            v2 = np.dot(vi, self.w[2]) + self.w[3]
            output.append(v2)
        fig = plt.figure()
        # plt.scatter(self.xn, self.dn, c='#cd0000')
        plt.plot(x, output)

    def plotBar(self):
        fig = plt.figure()
        plt.bar(range(len(self.errs)), self.errs, color='#045cff')

    def plotShow(self):
        plt.show()


def preProcess(timg,tlab):
    pimgs = []
    i = 0
    num = 60000#len(timg)
    for i in range(num):
        pic = timg[i].ravel()
        pimgs.append(pic)
    pimgs = np.array(pimgs).T
    # centrelize##
    numi = len(pimgs)
    for i in range(numi):
        aver = sum(pimgs[i])/num
        pimgs[i]  = (pimgs[i]- aver)/10
    ######eigen decomposition#########
    # Cor = np.dot(pimgs.T,pimgs)/5000
    # evals,evecs = np.linalg.eig(Cor)
    # evecs = evecs
    # pimgs = np.dot(pimgs,evecs)
    # np.savetxt("pro_data.txt", pimgs, fmt="%f", delimiter=",")
    # np.save('process',pimgs)
    return pimgs.T

def run():
    train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
    train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
    test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
    test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
    print 'load images and labels'
    train_images = preProcess(train_images,train_labels)
    P = neuronNetwork()
    w = P.training(train_images ,train_labels)
    P.plotBar()

    errors = P.testing(test_images,test_labels)
    print errors
    print (float(errors)/10000)
    P.plotShow()
    return w


run ()