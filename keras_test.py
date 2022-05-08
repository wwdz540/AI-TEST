import numpy as np
import  matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

np.random.seed(1337)
X = np.linspace(-1,1,200)
np.random.shuffle(X)
N = np.random.normal(0,0.5,(200,)) ##噪声
Y=0.5 *  X +  2 + N
#plt.scatter(X,Y)
#plt.show()

x_train,y_train=X[:160],Y[:160]

x_test,y_test = X[160:],Y[160:]
model = Sequential()
model.add(Dense(input_dim=1,units=1))

model.compile(loss='mse',optimizer='sgd')

print("traning ....................")
for step in range(301):
    cost = model.train_on_batch(x_train,y_train)
    if cost % 100 == 0 :
        print('\n train cost:',cost)


print("\ntest ........")
cost = model.evaluate(x_test,y_test,batch_size=40)
print('\ntest cost :',cost)

W,b = model.layers[0].get_weights()
print('Weight=',W,'\n biases=',b)
y_pred= model.predict(X)
plt.plot(X,y_pred)
plt.show()


