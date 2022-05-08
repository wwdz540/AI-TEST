from asyncore import dispatcher_with_send
import numpy as np
class RNN:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx), 
        np.zeros_like(Wh), 
        np.zeros_like(b)]

    
    def forward(self,x,h_prev):
        Wx , Wh = self.params
        t = np.dot(h_prev,Wh) + np.dot(x,Wh) + b
        h_next = np.tanh(t)
        
        self.cache = (x,h_prev,h_next)
        return h_next
    
    def backward(self,dh_next):
        Wx,Wh,b = self.params
        x,h_prev,h_next = self.cache

        dt = dh_next * (1-h_next ** 2)
        db = np.sum(dt,axis=0)
        dWh = np.dot(h_prev.T,dt)
        dh_prev=np.dot(dt,Wh.T)
        dWh = np.dot(x.T,dt)
        dx = np.dot(dt,Wx.T)

        self.grads[0][...]=dWx
        self.grads[1][...] =dWh


    