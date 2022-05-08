from cmath import sin
import os;
import  matplotlib.pyplot as plt 
import  numpy as np



if __name__ == '__main__':
    x  = np.linspace(10,20,100)
    print(x)
    plt.plot()
    plt.plot(x,np.sin(x))
    plt.show()