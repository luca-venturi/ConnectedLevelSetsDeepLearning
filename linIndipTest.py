# In this case there should be no bad local minima, assuming S=I

import tensorflow as tf
import numpy as np
from genCov import *

# model dimensions

n = 20
p = 1
m = 20

# model parameters

W = tf.Variable(tf.random_normal([p,n]))
U = tf.Variable(tf.random_normal([m,p]))

# model loss function

SX = tf.constant(computeSigmaX(n),dtype=tf.float32)
c = np.arange(n)
#SYX = tf.constant(computeSigmaYX(c,n),dtype=tf.float32)
SYX = tf.constant(np.random.normal(scale=2.0,size=(n,n)),dtype=tf.float32)

U_W = tf.matmul(U,W)
U_W_SX = tf.matmul(U_W,SX)
U_W_SX_W = tf.matmul(U_W_SX,tf.transpose(W))
U_W_SX_W_U = tf.matmul(U_W_SX_W,tf.transpose(U))
loss_pos = tf.trace(U_W_SX_W_U)

SYX_W = tf.matmul(SYX,tf.transpose(W))
SYX_W_U = tf.matmul(SYX_W,tf.transpose(U))
loss_neg = tf.trace(SYX_W_U)

loss = loss_pos - 2 * loss_neg

#sess = tf.Session()
#sess.run(loss)

# optimizer

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

# training/accuracy computing loop

init = tf.global_variables_initializer()
sess = tf.Session()
itTest = 10
itGradDesc = 10000
for _ in range(itTest):
	sess.run(init)
	for i in range(itGradDesc):
  		sess.run(train)
	
	WOpt, UOpt, lossOpt = sess.run([W, U, loss])
	print("minimum loss found: %s"%(lossOpt))
