# -*- coding: utf-8 -*-
import numpy as np
import random
import itertools

class Qlean():
    def __init__(self,n = 3,G = 2,rf = 0.9, arate = 0.6, epoch = 10,step = 40):
        self.n = n
        self.G = G
        self.initEnv()
        self.rf = rf
        self.arate = arate
        self.epoch = epoch
        self.step = step     
    
    def initloc(self):
        self.state = np.array([[1,1],0])
        
    def initEnv(self):
        self.initloc()
        self.q = np.random.randn((self.n*self.n - 2)*(self.G+1),7)
        g = range(0,self.G+1)
        row = range(1,self.n+1)
        col = range(1,self.n+1)
        label = itertools.product(row,col,g)
        qrow = 0
        for i in label:
            if (i[0:2] == (1,self.n)) or (i[0:2] == (self.n,1)):
                continue
            self.q[qrow][0] = i[0]
            self.q[qrow][1] = i[1]
            self.q[qrow][2] = i[2] 
            qrow = qrow + 1               
            
    def faction(self):
        for i in range(len(self.q)):
            if (self.q[i][0] == self.state[0][0]) and (self.q[i][1] == self.state[0][1]) and (self.q[i][2] == self.state[1]):
                action_list = [self.q[i][3],self.q[i][4],self.q[i][5],self.q[i][6]]                                            
                return i, action_list.index(max(action_list))
        
    def qaction(self):
        for i in range(len(self.q)):
#            print(i)
            if (self.q[i][0] == self.state[0][0]) and (self.q[i][1] == self.state[0][1]) and (self.q[i][2] == self.state[1]):
                action_list = [self.q[i][3],self.q[i][4],self.q[i][5],self.q[i][6]]               
                # random action 
                if np.random.rand(1) > 0.85:                                 
                    return i, np.random.randint(0,4)                             
                return i, action_list.index(max(action_list))                           
    
    def updateState(self,action1):
        # action up=[1,0], down = [-1,0], left = [0,-1], right = [0,1]
        if action1 == 0:
            action = [1,0]
        elif action1 == 1:
            action = [-1,0]
        elif action1 == 2:
            action = [0,-1]
        else:
            action = [0,1]          
        reward = 0
        nexloc = self.state[0] + np.array(action)
        # boundary
        if (nexloc[0] == 0) or (nexloc[0] == self.n+1) or (nexloc[1] == self.n+1) or (nexloc[1] == 0):
            reward = 0
        elif (nexloc[0] == self.n) and (nexloc[1] == 1) :#  nexloc == np.array([self.n,1]):
            if self.state[1] != 0:
                reward = self.state[1]
            self.state[1] = 0  
        elif ((nexloc[0] == 1) and (nexloc[1] == self.n-1)) or ((nexloc[0] == 2) and (nexloc[1] == self.n)):
            if self.state[1] < self.G:
                self.state[1] =  self.state[1] + 1
            self.state[0] = nexloc
        # plus mining
        elif (nexloc[0] == 1) and (nexloc[1] == self.n):
            if self.state[1] < self.G:
                self.state[1] =  self.state[1] + 1       
        else:
            self.state[0] = nexloc
        return reward     
     
    def getQvalue(self,action):
       for i in range(len(self.q)):
            if (self.q[i][0] == self.state[0][0]) and (self.q[i][1] == self.state[0][1]) and (self.q[i][2] == self.state[1]):
                if action == 'max':
                    return max(self.q[i][3],self.q[i][4],self.q[i][5],self.q[i][6])
                else:
                    return self.q[i][action + 3]
                break
        
    def updateQ(self,i,action,r,Qn):
        self.q[i][action + 3] = (1-self.arate)*self.q[i][action + 3] + self.arate*(r + self.rf*Qn)
        
    def training(self):      
        for ep in range(self.epoch):
            self.initloc()
            rc = 0
            for i in range(self.step):              
                old_state, action = self.qaction()
                r = self.updateState(action) # reward while motion
                Qn = self.getQvalue('max') # return next Qmax
                self.updateQ(old_state,action,r,Qn)
        
    def testing(self):
        self.initloc()
        rc = 0
        rs = 0
        steps = []
        for i in range(self.step):              
            # pick an action
            old_state, action = self.faction()

            print(self.state)
            steps.append(actiondic[action])

            r = self.updateState(action) # reward while motion
            print(i,r)
            rs = rs + r
            rc = rc + r*pow(self.rf,i+1)#self.rf
        print(steps)
        print(rc,rs)
    
actiondic = {0:'U',1:'D',2:'L',3:'R'}   
# self,n = 3,G = 2,rf = 0.9, arate = 0.6, epoch = 10,step = 40
test = Qlean(5,3,0.6,1,1000,60)
test.training()
test.testing()


        