import numpy as np
import random
import matplotlib.pyplot as plt



def linear(x,y):
    for i in range(len(x)):
        y[i] = x[i] -1 + 2*random.random()
    avx = float(sum(x))/float(len(x))
    avy = float(sum(y))/float(len(y))
    avxy = float(sum(x*y))/float(len(x))
    avxx = float(sum(x*x))/float(len(x))
    w1 = (avxy - avx*avy)/(avxx - avx*avx)
    w0 = avy - w1*avx
    xp = np.linspace(0, 51, 100)
    yp = w0 + w1*xp
    plt.plot(xp, yp)
    plt.scatter(x,y,c='#DDCB00')

def gradi(x,y):
    a = 0.000001
    w0 = 0.1
    w1 = 0.8
    print ("graaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    for i in range(300000):
        s0 = 0
        s1 = 0
        for k in range(len(x)):
            s0 = s0 + (-y[k] + (w0 + w1*x[k]))
            s1 = s1 + (-y[k] + (w0 + w1*x[k]))*x[k]
        t0 = w0 - a*2*s0
        t1 = w1 - a*2*s1
        w0 = t0
        w1 = t1
        print("w0 = ",w0,"  w1 = ", w1)
    xp = np.linspace(0, 51, 100)
    yp = w0 + w1 * xp
    plt.plot(xp, yp)

random.seed(0)
x = np.arange(50)+1
y = np.zeros(50)

linear(x,y)
gradi(x,y)

plt.legend(['Analytical','Gradient Descent','Points'])
plt.show()