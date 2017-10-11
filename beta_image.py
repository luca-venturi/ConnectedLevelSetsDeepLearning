import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def relu(x):
	return np.maximum(np.zeros(x.shape),x)

n = 10
N = 100
data = np.random.normal(size=(N,n))

nRange = 100000
plotRange = np.random.normal(size=(n,nRange))

plotData = np.dot(data,plotRange)
#plotData = np.square(plotData) # quadratic kernel
plotData = relu(plotData) # relu kernel


plt.plot(plotData[0,:],plotData[1,:],'.')
plt.show()

'''
datasq = np.dot(data,data.T)
k = np.dot(data,data.T) # (N,N)
k = np.square(k)
k_inv = np.linalg.pinv(k)

nRange = 100000
plotRange = np.random.normal(size=(n,nRange))
kRange = np.dot(data,plotRange) # (N,nRange)
kRange = np.square(kRange)

plotData = np.dot(k_inv,kRange) # (N,nRange)
print plotData.shape

plt.plot(plotData[0,:],plotData[1,:],'.')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(plotData[0,:],plotData[1,:], plotData[2,:])
plt.show()


### ReLU

n = 100
N = 10
data = np.random.normal(size=(N,n))
datasq = np.dot(data,data.T)
k = np.dot(data,data.T) # (N,N)
for i in range(N):
	for j in range(N):
		k[i,j] = relu(k[i,j])
k_inv = np.linalg.pinv(k)

nRange = 100000
plotRange = np.random.normal(size=(n,nRange))
kRange = np.dot(data,plotRange) # (N,nRange)
for i in range(N):
	for j in range(nRange):
		kRange[i,j] = relu(kRange[i,j])

plotData = np.dot(k_inv,kRange) # (N,nRange)
print plotData.shape

plt.plot(plotData[0,:],plotData[1,:],'.')
plt.show()
'''
