This repo is where I'll house everything related to Machine Learning.  I started by
doing tutorials with TensorFlow but have quickly been sucked in by articles and
papers about the math behind it.

## NOTEBOOKS

### 1. [/Non-Tensorflow]({{ site.baseurl }}/nn_backprop_notebook.html)
This contains the code for a deep neural net (variable number of layers, nodes per layer)
that I adapted from a shallow neural net implementation [here](https://medium.com/@curiousily/tensorflow-for-hackers-part-iv-neural-network-from-scratch-1a4f504dfa8), and it is completely
done without TensorFlow.  It uses NumPy for the linear algebra and matplot for
visualization.

I have been reading about optimizing the hyperparameters of a deep neural net since
they are notoriously difficult to train, and after a lot of testing, I settled on
an accuracy of 98.4% with my deep NN implementation.  This should give me a solid
base to maximize the accuracy of a TF estimator on a dataset I pulled from the UCI
Machine Learning Repository.

### 2. [/DeepNN]({{ site.baseurl }}/TensorFlowDeepNN.html)
This contains the code for a deep neural net, but this time it is implemented using
TensorFlow's low-level API.  This means that data is manually parsed from the
[official MNIST website](http://yann.lecun.com/exdb/mnist/), the network is
defined by manually initialized and connected tensors, and the step function
is also defined manually.  The only magical part is the polynomial decay of the
learning rate, which is provided by TF.

This network is slightly faster to train, and achieves close to 98% accuracy in only
30 epochs (versus the gigantic 500 epochs for the previous notebook).  I could also
experiment with different pre-built estimators but of course the point of this
notebook was to better understand the low-level API.

### 3. [/VisualData]({{ site.baseurl }}/VisualizingData.html)
This notebook utilizes primarily matplotlib to visualize the UCI
drug consumption data.  While the depth of exploration is relatively shallow,
it is only meant to provide some intuition about the data in order to help
in building a better classifier, since of course the data should influence many
design decisions.

### 4. [/RealLearning]({{ site.baseurl }}/RealLearning.html)
This puts all the previous pieces together.  I use
the neural network developed in Part 2 as a base and try to achieve the best
accuracy that I can using personality traits and other features to predict
drug usage.

While Parts 1 & 2 had an explicit focus on not using pre-packaged TF tools, this
notebook aims to take advantage of anything available to boost accuracy to more
closely mirror a real application of the technology (albeit at a relatively small scale).