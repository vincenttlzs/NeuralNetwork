import random
import numpy as np
import matplotlib.pyplot as pplt

class vectorsGeo:
    def __init__(self,n = 100):
        self.n = n
        self.w0 = random.uniform(-0.25, 0.25)
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.colors = ['#002356','#DDCB00']
        self.set = ['S1','S0']
        self.vectors = self.generVector()

    def outputW(self):
        wp = [self.w0, self.w1, self.w2]
        return wp

    def generVector(self):
        a = np.array([-1+2*random.random(),-1+2*random.random()])
        for i in range(self.n):
            b = np.array([-1+2*random.random(),-1+2*random.random()])
            a = np.vstack((a,b))
        return a

    def classify(self):
        s1 = []
        s0 = []
        for i in range(self.n):
            if(self.w0+self.w1*self.vectors[i,0]+self.w2*self.vectors[i,1] >= 0):
                s1.append(self.vectors[i])
            else:
                s0.append(self.vectors[i])
        return s1,s0

    def plotPoint(self,arr,no):
        x = []
        y = []
        for i in range(len(arr)):
            x.append(arr[i][0])
            y.append(arr[i][1])
        pplt.scatter(x,y,c=self.colors[no])

    def drawLine2(self,w0,w1,w2):
        x = np.linspace(-1, 1, 100)
        y = (-w0 - w1 * x) / w2
        pplt.plot(x,y)

    def plotShow(self):
        pplt.show()

    def neuralNetwork(self,s1,s0,w0p = -0.5,w1p = -0.1,w2p = 0.8,ny = 1):
        misClssNum = []
        while(1):
            misNum = 0
            for i in range(len(s1)):
                if(1*w0p + s1[i][0]*w1p + s1[i][1]*w2p >=0):
                    pass
                else:
                    misNum += 1
            for i in range(len(s0)):
                if (1 * w0p + s0[i][0] * w1p + s0[i][1] * w2p < 0):
                    pass
                else:
                    misNum += 1
            misClssNum.append(misNum)
            if misNum == 0:
                break
            for i in range(len(s1)):
                if(1*w0p + s1[i][0]*w1p + s1[i][1]*w2p >=0):
                    pass
                else:
                    w0p = w0p + ny*1
                    w1p = w1p + ny * s1[i][0]
                    w2p = w2p + ny * s1[i][1]
            for i in range(len(s0)):
                if (1 * w0p + s0[i][0] * w1p + s0[i][1] * w2p < 0):
                    pass
                else:
                    w0p = w0p - ny * 1
                    w1p = w1p - ny * s0[i][0]
                    w2p = w2p - ny * s0[i][1]

            wp = [w0p, w1p, w2p]
        return wp,misClssNum

# Question (i)
fig = pplt.figure()
po = vectorsGeo(100) # you can show the result when sample = 1000 through displacing the 100 with 1000
weight_orig = po.outputW()
print("The original weigh: ",weight_orig)
up,down = po.classify()
po.plotPoint(up,0)
po.plotPoint(down,1)
po.drawLine2(po.outputW()[0],po.outputW()[1],po.outputW()[2])
pplt.xlabel('X')
pplt.ylabel('Y')
pplt.ylim([-1.1,1.1])
pplt.xlim([-1.1,1.1])
pplt.legend(['Boundary','S1','S0'])

# Question (j)
w = [-0.4,-0.3,0.9]
print("The random beginning weigh of network: ",w)
ws1, mis1 = po.neuralNetwork(up,down,w[0],w[1],w[2],1)
ws2, mis2 = po.neuralNetwork(up,down,w[0],w[1],w[2],10)
ws3, mis3 = po.neuralNetwork(up,down,w[0],w[1],w[2],0.1)
fig = pplt.figure()
po.plotPoint(up,0)
po.plotPoint(down,1)
pplt.legend(['S1','S0'])
po.drawLine2(po.outputW()[0],po.outputW()[1],po.outputW()[2])
po.drawLine2(ws1[0],ws1[1],ws1[2])
pplt.ylim([-1.1,1.1])
pplt.xlim([-1.1,1.1])
pplt.legend(['Boundary',' Learned Boundary','S1','S0'])
print("The final weigh when ny = 1: ",ws1)
print("Misclassifications when ny = 1: ",mis1)
print("Number of misclassifications when ny = 1: ",len(mis1))
print("\n")
print("The final weigh when ny = 10: ",ws2)
print("Misclassifications when ny = 10: ",mis2)
print("Number of misclassifications when ny = 10: ",len(mis2))
print("\n")
print("The final weigh when ny = 0.1: ",ws3)
print("Misclassifications when ny = 0.1: ",mis3)
print("Number of misclassifications when ny = 0.1: ",len(mis3))

# Question (k~m)
fig = pplt.figure()
fig.add_subplot(2,2,1)
pplt.bar(range(len(mis1)),mis1, color = '#00ddca')
pplt.xlabel('Epoch')
pplt.ylabel('Num of Misclassifications')
pplt.title('Value ny = 1')
fig.add_subplot(2,2,2)
pplt.bar(range(len(mis2)),mis2, color = '#f9ddd2')
pplt.xlabel('Epoch')
pplt.ylabel('Num of Misclassifications')
pplt.title('Value ny = 10')
fig.add_subplot(2,2,3)
pplt.bar(range(len(mis3)),mis3, color = '#045cff')
pplt.xlabel('Epoch')
pplt.ylabel('Num of Misclassifications')
pplt.title('Value ny = 0.1')

po.plotShow()

#############################################################################################################
#***************************************  THE OUTPUT OF A RUN **********************************************#
#############################################################################################################
# ('The original weigh: ', [-0.19702088433097742, 0.0008518456979891287, 0.6553600805481203])
# ('The random beginning weigh of network: ', [-0.4, -0.3, 0.9])
# ('The final weigh when ny = 1: ', [-1.4, 0.0822574198701298, 4.4074862663634127])
# ('Misclassifications when ny = 1: ', [12, 27, 18, 10, 7, 23, 3, 17, 0])
# ('Number of misclassifications when ny = 1: ', 9)
# 
# 
# ('The final weigh when ny = 10: ', [-30.4, 1.5628272317846861, 93.687997410498468])
# ('Misclassifications when ny = 10: ', [12, 28, 16, 31, 25, 15, 16, 13, 9, 7, 4, 5, 4, 14, 3, 3, 10, 9, 9, 4, 8, 4, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 9, 9, 9, 9, 8, 0])
# ('Number of misclassifications when ny = 10: ', 38)
# 
# 
# ('The final weigh when ny = 0.1: ', [-0.30000000000000004, -0.0068409971632727926, 0.99216478041295275])
# ('Misclassifications when ny = 0.1: ', [12, 6, 5, 0])
# ('Number of misclassifications when ny = 0.1: ', 4)
#############################################################################################################


