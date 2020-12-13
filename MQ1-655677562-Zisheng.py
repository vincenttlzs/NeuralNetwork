import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class competition(object):
    def __init__(self,img,devide,k,epoch,ny):
        self.img = np.array(img)
        self.epoch = epoch
        self.ny = ny
        self.d = devide
        self.k = k
        self.slice()
        self.neuronInit()

    def slice(self):
        if (512 % self.d != 0):
            return 1
        n = 512 / self.d
        A = []
        Aij = []
        for i in range(n):
            for j in range(n):
                Aij.append(i)
                Aij.append(j)
                a = np.zeros((self.d, self.d))
                for d1 in range(self.d):
                    for d2 in range(self.d):
                        a[d1, d2] = self.img[d1 + i * self.d, d2 + j * self.d]
                Aij.append(a)
                A.append(Aij)
                Aij = []
        self.A = A

    def neuronInit(self):
        X = []
        for i in range(self.k):
            X.append(self.A[1][2])
        self.X = X

    def training(self):
        d = 2
        ks = []
        rmss = []
        AP = np.zeros((self.img.shape[0], self.img.shape[1]))
        for epoch in range(self.epoch):
            for ij in range(len(self.A)):
                for ik in range(len(self.X)):
                    AX = np.linalg.norm(self.A[ij][2] - self.X[ik])
                    ks.append(AX*AX)
                index = ks.index(min(ks))
                self.X[index] = (1-self.ny)*self.X[index] + self.ny*self.A[ij][2]
                ks = []
                for i in range(self.X[index].shape[0]):
                    for j in range(self.X[index].shape[1]):
                        AP[self.d*self.A[ij][0]+i,self.d*self.A[ij][1]+j] = self.X[index][i,j]
            print ("RMS: ",(np.linalg.norm(self.img - AP))/512)
            rmss.append((np.linalg.norm(self.img - AP))/512)
        # fig = plt.figure()
        plt.bar(range(len(rmss)), rmss, color='#045cff')


    def testing(self):
        AP = np.zeros((self.img.shape[0],self.img.shape[1]))
        ks = []
        for ij in range(len(self.A)):
            for ik in range(len(self.X)):
                AX = np.linalg.norm(self.A[ij][2] - self.X[ik])
                ks.append(AX * AX)
            index = ks.index(min(ks))
            ks = []
            for i in range(self.X[index].shape[0]):
                for j in range(self.X[index].shape[1]):
                    AP[self.d*self.A[ij][0]+i,self.d*self.A[ij][1]+j] = self.X[index][i,j]
        pic2 = Image.fromarray(AP)
        pic2.show()
        #pic2.save('ss2.bmp')


im = Image.open('./barbara.png')
# im_a = np.array(im)
# pic = Image.fromarray(im_a)
# pic.show()

com = competition(im,4,8,10,0.1) # Input:image,division,K,epoch,learning rate
com.training()
com.testing()
plt.show()




