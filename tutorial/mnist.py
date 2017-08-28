from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Placeholder for input, W and b are trained
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# This normalizes the distribution
# y is a vector holding probabilities of x being each digit
y = tf.nn.softmax(tf.matmul(x, W) + b)

# To define the loss, use y* as expected distribution
# and use cross_entropy formula
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Use gradient descent to minimize cross_entropy loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Initializing stuff
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Run many learning iterations
# mnist gives batches of xs and ys and the session runs every one in the batch
# to train the W and b variables
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# argmax will find the highest probability entry in y, which
# is the predicted digit in y and the correct digit in y*
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# since tf.equal means correct_prediction is a bool vector,
# we cast to floats and give the average (which will be num_correct/total)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# this just prints the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))