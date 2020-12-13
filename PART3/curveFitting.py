import numpy as np
import math
import random
import matplotlib.pyplot as plt


class neuronNetwork():
    def __init__(self,neuNumList= [24,24,24,1],ny = 0.1):
        self.neuNumList = neuNumList
        self.neuronInit()
        self.ny = ny
        self.msess = []

    def neuronInit(self):
        w = []
        for i in range(len(self.neuNumList)):
            wi = np.random.rand(self.neuNumList[i])
            w.append(wi)
        self.w = w

    def training(self,x,dn):
        for e in range(10):
            mse = []
            for n in range(len(xn)):
                temp =  self.w[1] + np.dot(xn[n], self.w[0])
                vi = []
                for j in range(temp.shape[0]):
                    y = math.tanh(temp[j])
                    vi.append(y)
                v2 = np.dot(vi,self.w[2]) + self.w[3]
                mse.append((dn[n] - v2)*(dn[n] - v2))
            mses = sum(mse)/len(xn)
            self.msess.append(mses)
            print mses
            if self.ny >=0.2:
                self.ny = self.ny*0.95

            for e1 in range(150):
                for n in range(len(xn)/4):
                    self.updateW(dn,n)

            for e2 in range(150):
                for n in range(len(xn)/4):
                    n = n + len(xn)/4
                    self.updateW(dn,n)

            for e3 in range(100):
                for n in range(len(xn)/4):
                    n = n + 2 * len(xn) / 4
                    self.updateW(dn,n)

            for e4 in range(150):
                for n in range(len(xn)/4):
                    n = n + 3 * len(xn)/4
                    self.updateW(dn,n)

    def updateW(self,dn,n):
        temp = self.w[1] + np.dot(xn[n], self.w[0])
        vi = []
        for j in range(temp.shape[0]):
            y = math.tanh(temp[j])
            vi.append(y)
        v2 = np.dot(vi, self.w[2]) + self.w[3]
        dd = dn[n] - v2
        for ii in range(len(self.w[1])):
            t1 = self.w[0][ii] + self.ny * xn[n] * (dd) * self.w[2][ii] * self.tanhp(temp[ii])
            t2 = self.w[1][ii] + self.ny * 1 * (dd) * self.w[2][ii] * self.tanhp(temp[ii])
            t3 = self.w[2][ii] + self.ny * vi[ii] * (dd)
            self.w[0][ii] = t1
            self.w[1][ii] = t2
            self.w[2][ii] = t3
        self.w[3] = self.w[3] + self.ny * dd

    def tanhp(self,v):
        return 1 - pow(math.tanh(v),2)

    def plotCurve(self):
        x = np.linspace(0, 1, 100)
        output = []
        for n in range(len(x)):
            temp = self.w[1] + np.dot(x[n], self.w[0])
            vi = []
            for j in range(temp.shape[0]):
                y = math.tanh(temp[j])
                vi.append(y)
            v2 = np.dot(vi, self.w[2]) + self.w[3]
            output.append(v2)
        fig = plt.figure()
        plt.scatter(xn, dn, c='#cd0000')
        plt.plot(x, output)

    def plotBar(self):
        fig = plt.figure()
        plt.bar(range(len(self.msess)), self.msess, color='#045cff')

    def plotShow(self):
        plt.show()


random.seed(0)
n = 300
xn = np.random.rand(n)
vn = (np.random.rand(n))*0.2 - 0.1
dn = []
for i in range(n):
    dn.append(math.sin(20*xn[i]) + 3*xn[i] + vn[i])
dn = np.array(dn)

t = neuronNetwork()
t.training(xn,dn)
t.plotCurve()
t.plotBar()
t.plotShow()




