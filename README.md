# tensorflow_stuff
Getting acquainted with TensorFlow.

This repo is where I'll house everything related to Machine Learning.  I started by
doing tutorials with TensorFlow but have quickly been sucked in by articles and
papers about the math behind it.  There is a Jupyter notebook in this repo that
contains the code for a deep neural net (variable number of layers, nodes per layer)
that I adapted from a shallow neural net implementation, and it is completely
done without TensorFlow.  It uses NumPy for the linear algebra and matplot for
visualization.

I have been reading about optimizing the hyperparameters of a deep neural net since
they are notoriously difficult to train, and my next goal is to push test accuracy
for MNIST past 98.5% with my deep NN implementation.  This should give me a solid
base to maximize the accuracy of a TF estimator on a dataset I pulled from the UCI
Machine Learning Repository.

Long-term (~1 month) I aim to deploy and host some sort of visualization of machine
learning on a real dataset to showcase what I have learned.
