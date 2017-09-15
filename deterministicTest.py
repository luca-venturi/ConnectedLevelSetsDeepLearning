import tensorflow as tf
import numpy as np

# model dimensions

n = 3
p = 2
m = 1

# model parameters

W = tf.Variable(tf.random_normal([n, p])) #
b = tf.Variable(tf.random_normal([p, m])) #

# model input and output

x = tf.placeholder(tf.float32, shape=(n, 1))
dnn_model = tf.transpose(tf.matmul(tf.square(tf.matmul(tf.transpose(x), W)),b))
y = tf.placeholder(tf.float32, shape=(m,1))

# loss

loss = tf.reduce_sum(tf.square((dnn_model - y))) # this should be the only thing I have to change in the other alg. variables x and y would now become matrices

# optimizer

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data

x_train = np.random.normal(size=(n,1)) #
y_train = np.random.normal(size=(m,1)) #

# training/accuracy computing loop

init = tf.global_variables_initializer()
sess = tf.Session()
itTest = 10
itGradDesc = 1000
for _ in range(itTest):
	sess.run(init)
	for i in range(itGradDesc):
  		sess.run(train, {x: x_train, y: y_train})
	
	WOpt, bOpt, lossOpt = sess.run([W, b, loss], {x: x_train, y: y_train})
	print("minimum loss found: %s"%(lossOpt))
