# In this case there should be no bad local minima, is S = I

import tensorflow as tf
import numpy as np
from genCov import *

# model dimensions

n = 2
p = 4
m = 1

# model parameters

W = tf.Variable(tf.random_normal([1,p,n]))
U = tf.Variable(tf.random_normal([m,p]))
W0 = tf.Variable(tf.random_normal([1,p,n]))
U0 = tf.Variable(tf.random_normal([m,p]))
copyW = W0.assign(W)
copyU = U0.assign(U)
copyW0 = W.assign(W0)
copyU0 = U.assign(U0)

# model loss function

W2 = tf.tensordot(W,W,[[0],[0]])
#delta = tf.placeholder(tf.float32, shape=(p,p,p))
#Delta = computeDelta(p)
delta = tf.constant(computeDelta(p),dtype=tf.float32)
#SX2 = tf.placeholder(tf.float32, shape=(n,n,n,n))
#SigmaX2 = computeSigmaX2(n)
SX2 = tf.constant(computeSigmaX2(n),dtype=tf.float32)
#SYX2 = tf.placeholder(tf.float32, shape=(m,n,n))
#SigmaYX2 = computeSigmaYX2(c,n)
SigmaYX2 = [[[2,0],[0,-1]]]
SYX2 = tf.constant(SigmaYX2,dtype=tf.float32)

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

#optimizer = tf.train.GradientDescentOptimizer(0.001)
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)
optimizer1 = tf.train.AdamOptimizer(0.001)
train1 = optimizer1.minimize(loss)

# training/accuracy computing loop

init = tf.global_variables_initializer()
sess = tf.Session()
itTest = 10
itGradDesc = 20000
itGradDescIncrease = 5000
for _ in range(itTest):
	sess.run(init)
	sess.run(copyW)
	sess.run(copyU)
	for i in range(itGradDesc):
  		sess.run(train)
	
	WOpt, UOpt, lossOpt = sess.run([W, U, loss])
	print("minimum loss found: %s"%(lossOpt))

	sess.run(copyW0)
	sess.run(copyU0)
	for i in range(itGradDesc+itGradDescIncrease):
  		sess.run(train1)
	
	WOpt, UOpt, lossOpt = sess.run([W, U, loss])
	print("minimum loss found (train1): %s"%(lossOpt))
