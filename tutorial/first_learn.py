import tensorflow as tf

# variables and init
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
init = tf.global_variables_initializer()

# placeholder and model
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# loss measurements
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# run init
sess = tf.Session()
sess.run(init)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# run training
for i in range(1000):
	sess.run(train, {x: [1,2,3,4], y: [0,-1,-2,-3]})

print(sess.run([W,b]))