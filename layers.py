import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1,keepdims = True)
        x = np.exp(x)
        x /= x.sum(axis=1,keepdims = True)
    elif x.ndim == 1:
        x = x - x.sum()
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def relu(x):
    return np.maximum(0,x)



def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-7))/batch_size

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self,x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class MatMul:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self,x):
        W, = self,params
        out = np.dot(x,W)
        self.x = x
        return out
    
    def backward(self,dout):
        W,=self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        self.grads[0][...] =dW
        return dx



class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class Softmax:
    def __init__(self):
        self.params,self.grads = [],[]
        self.out= None
    
    def forward(self,x):
        self.out = softmax(x)
        return self.out
    
    def backward(self,dout):
        dx = self.out*dout
        sumdx = self.out*sumdx
        dx -= self.out*sumdx
        return dx
    

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [],[]
        self.y = None
        self.x = None
    
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):

        batch_size = self.t.shape[0]
        if(self.t.size == self.y.size):
            dx = (self.y - self.t) /batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size),self.t] -=1
            dx = dx/batch_size
        return dx



class Sigmoid:
    def __init__(self):
        self.params ,self.grads = []
        self.out = None

    def forward(self , x):
        out = 1/(1+np.exp(-1))
        self.out= out
        return out

    def backward(self,out):
        dx = dount * (1.0 - self.out) * self.out
        return dx

class SigmoidWithLoss:
    def __init__(self):
        self.params,self,grads = [],[]
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self,x,t):
        self.t = t
        self.y = 1/(1+np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1-self.y,self.y],self.t)
        return self.loss
    
    def backward(self,dount=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) * dount /batch_size
        return dx
    

    

    
    
