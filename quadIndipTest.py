# In this case there should be no bad local minima, assuming S=I

import tensorflow as tf
import numpy as np
from genCov import *

# model dimensions

n = 3
p = 2
m = 1

# model parameters

W = tf.Variable(tf.random_normal([1,p,n]))
U = tf.Variable(tf.random_normal([m,p]))

# model loss function

W2 = tf.tensordot(W,W,[[0],[0]])
#delta = tf.placeholder(tf.float32, shape=(p,p,p))
#Delta = computeDelta(p)
delta = tf.constant(computeDelta(p),dtype=tf.float32)
#SX2 = tf.placeholder(tf.float32, shape=(n,n,n,n))
#SigmaX2 = computeSigmaX2(n)
SX2 = tf.constant(computeSigmaX2(n),dtype=tf.float32)
#SYX2 = tf.placeholder(tf.float32, shape=(m,n,n))
c = np.ones((m))
#SigmaYX2 = computeSigmaYX2(c,n)
SYX2 = tf.constant(computeSigmaYX2(c,n),dtype=tf.float32)

U_delta = tf.tensordot(U,delta,[[1],[0]])
U_delta_W2 = tf.tensordot(U_delta,W2,[[1,2],[0,2]])
U_delta_W2_SX2 = tf.tensordot(U_delta_W2,SX2,[[1,2],[0,1]])
U_delta_W2_SX2_W2 = tf.tensordot(U_delta_W2_SX2,W2,[[1,2],[1,3]])
U_delta_W2_SX2_W2_delta = tf.tensordot(U_delta_W2_SX2_W2,delta,[[1,2],[0,1]])
U_delta_W2_SX2_W2_delta_U = tf.tensordot(U_delta_W2_SX2_W2_delta,U,[[1],[1]])
loss_pos = tf.trace(U_delta_W2_SX2_W2_delta_U)

SYX2_W2 = tf.tensordot(SYX2,W2,[[1,2],[1,3]])
SYX2_W2_delta = tf.tensordot(SYX2_W2,delta,[[1,2],[0,1]])
SYX2_W2_delta_U = tf.tensordot(SYX2_W2_delta,U,[[1],[1]])
loss_neg = tf.trace(SYX2_W2_delta_U)

loss = loss_pos - 2 * loss_neg

# optimizer

optimizer = tf.train.GradientDescentOptimizer(0.01)
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
