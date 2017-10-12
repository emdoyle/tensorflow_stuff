---
layout: notebook
---
# Using TensorFlow to Predict Drug Usage

In this notebook I will be building an Estimator using tf.estimator to classify drug usage in the [UCI dataset](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29) and comparing my results to the accompanying [paper](https://arxiv.org/abs/1506.06297).  The paper will also give me some insight into how to customize the Estimator to each classification task.


```python
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import constants

from utils import compare_results, process_column_names, one_hot
from tensorflow.contrib.learn.python.learn.datasets import base

# Less Verbose Output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

DATA_DIR = "Dataset/"
DRUG_TRAINING = DATA_DIR + "re_drug_consumption_data.csv"
DRUG_TEST = DATA_DIR + "test_drug_consumption_data.csv"
DRUG_PREDICT = DATA_DIR + "predict_drug_consumption_data.csv"

PREDICT_OUTPUT = DATA_DIR + "predictions.csv"

NUM_CLASSES = 2

# Bucketization for possible nonlinear relationship
age = tf.feature_column.numeric_column("age")
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=constants.AGE_BOUNDARIES)
gender = tf.feature_column.numeric_column("gender")
gender_buckets = tf.feature_column.bucketized_column(
    gender, boundaries=constants.GENDER_BOUNDARIES)
country = tf.feature_column.numeric_column("country")
country_buckets = tf.feature_column.bucketized_column(
    country, boundaries=constants.COUNTRY_BOUNDARIES)
ethnicity = tf.feature_column.numeric_column("ethnicity")
ethnicity_buckets = tf.feature_column.bucketized_column(
    ethnicity, boundaries=constants.ETHNICITY_BOUNDARIES)
education = tf.feature_column.numeric_column("education")
education_buckets = tf.feature_column.bucketized_column(
    education, boundaries=constants.EDUCATION_BOUNDARIES)

# Could bucketize but guessing these are close to linear
nscore = tf.feature_column.numeric_column("nscore")
escore = tf.feature_column.numeric_column("escore")
oscore = tf.feature_column.numeric_column("oscore")
ascore = tf.feature_column.numeric_column("ascore")
cscore = tf.feature_column.numeric_column("cscore")
impulsive = tf.feature_column.numeric_column("impulsive")
ss = tf.feature_column.numeric_column("ss")

nscore_ascore = tf.feature_column.crossed_column(
    ["nscore", "ascore"], hash_bucket_size=500)
nscore_cscore = tf.feature_column.crossed_column(
    ["nscore", "cscore"], hash_bucket_size=500)
ascore_cscore = tf.feature_column.crossed_column(
    ["ascore", "cscore"], hash_bucket_size=500)
nscore_ascore_cscore = tf.feature_column.crossed_column(
    ["nscore", "ascore", "cscore"], hash_bucket_size=500)
```


```python
def input_fn(data_file, target, num_epochs, feature_columns, batch_size=30, shuffle=False, num_threads=1):
    dataset = pd.read_csv(
        tf.gfile.Open(data_file),
        header=0,
        usecols=feature_columns + [target],
        skipinitialspace=True,
        engine="python")
    # Drop NaN entries
    dataset.dropna(how="any", axis=0)

    # Init empty dataframe, add column for each of targets
    labels = pd.DataFrame(columns=[target])
    
    # This assigns a different number to each usage category
    # labels[constants.TARGET] = dataset[constants.TARGET].apply(lambda x: constants.MAPPED_CODES[x]).astype(int)

    # This classifies usage as binary (USER/NON-USER) to make prediction easier
    labels[target] = dataset[target].apply(lambda x: x in constants.USER).astype(int)
    dataset.pop(target)
    
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(dataset)},
        y=np.array(one_hot(NUM_CLASSES, labels[target])),
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1)
```


```python
# NOTE: This is a hacky solution due to the following issue:
# the tf.layers weights are called dense_[number]/kernel:0
# but while tf.GraphKeys.TRAINABLE_VARIABLES records dense/kernel:0, dense_1/kernel:0...
# layer.name for some reason records dense/kernel:0, dense_2/kernel:0...
# So I need to decrement the number on the end of 'dense' if there is one.
def extract_weights(layer):
    name = os.path.split(layer.name)[0]
    if '_' in name:
        number = str(int(name[name.find('_')+1:]) - 1)
        name = name[:name.find('_')] + '_' + number
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        if variable.name == name + '/kernel:0':
            return variable
    return None
```


```python
def model_fn(features, labels, mode, params):
    
    # 1. Configure the model via TensorFlow operations
    input_layer = tf.cast(features["x"], tf.float32)
    
    beta = params["beta"]
    layer_sizes = params["hidden_layers"]
    current_tensor = input_layer
    regularizer = tf.contrib.layers.l2_regularizer(beta)
    weights = []
    for nodes in layer_sizes:
        current_tensor = tf.layers.dense(current_tensor, nodes, activation=tf.nn.tanh,
            kernel_regularizer=regularizer)
        weights.append(extract_weights(current_tensor))
        
    output_layer = tf.layers.dense(current_tensor, 2, activation=tf.nn.tanh)

    # 4. Generate predictions
    predictions = output_layer

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"usage": predictions})
    
    # 2. Define the loss function for training/evaluation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels))
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    loss += reg_term

    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(tf.cast(labels,tf.float64), tf.cast(predictions,tf.float64)),
        "accuracy": tf.metrics.accuracy(
            tf.cast(tf.argmax(labels,1), tf.float64), tf.cast(tf.argmax(predictions,1), tf.float64))
    }
    
    # 3. Define the training operation/optimizer
    decay_steps = 100000
    learning_rate = tf.train.polynomial_decay(params["start_learn"], tf.train.get_global_step(),
                                          decay_steps, params["end_learn"],
                                          power=0.5)
    optimizer=tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
```


```python
def train_and_eval(classifier, target, feature_cols, batch_size=25, steps=100000):
    classifier.train(input_fn=input_fn(DRUG_TRAINING, feature_columns=feature_cols, target=target,
        batch_size=batch_size, num_epochs=None, shuffle=True), steps=steps)

    results = classifier.evaluate(input_fn=input_fn(DRUG_TEST, target=target, num_epochs=1,
        feature_columns=feature_cols, shuffle=False), steps=None)

    print(results)
    print("Accuracy: %.2f%%" % (results['accuracy']*100))
```


```python
def custom_predict(classifier, target, feature_cols, cutoff=0):
    predictions = classifier.predict(input_fn=input_fn(DRUG_PREDICT, target=target, num_epochs=1,
        feature_columns=feature_cols, shuffle=False))
    predict_writer = open(PREDICT_OUTPUT, "w")
    predict_writer.write("Fake header\n")
    for prediction in list(predictions):
        val1 = prediction["usage"][0]
        val2 = prediction["usage"][1]
        if val1 > val2:
            predict_writer.write('0\n')
        else:
            predict_writer.write('1\n')
    predict_writer.close()
```


```python
feature_columns = [
    age_buckets, gender_buckets, country_buckets, ethnicity_buckets,
    education_buckets, tf.feature_column.embedding_column(nscore_ascore_cscore, 1),
    nscore, escore, oscore, ascore, cscore, impulsive, ss
]

feature_col_names = process_column_names(feature_columns)

model_params = {
    "feature_columns": feature_columns,
    "hidden_layers": [10, 5],
    "start_learn": 0.15,
    "end_learn": 0.01,
    "beta": 0.001
}

nn = tf.estimator.Estimator(model_fn, params=model_params)
train_and_eval(nn, "cannabis", feature_col_names)
custom_predict(nn, "cannabis", feature_col_names)
compare_results(["cannabis"])
```

    {'accuracy': 0.8433333, 'loss': 0.40682358, 'rmse': 0.86066353, 'global_step': 100000}
    Accuracy: 84.33%
    Comparing 200:200
    cannabis Accuracy: 73.500000%
    cannabis Sensitivity: 72.566372%
    cannabis Specificity: 74.712644%


This estimator runs 100,000 training steps pretty quickly (about a minute and a half on my laptop), and has achieved better accuracy than we have seen so far on this dataset.  A few choices that I made with this model are:

1. Used polynomial decay to decay the learning rate.
  * This slows the learning rate as the model sees more examples to avoid overshooting a local loss minimum
2. Used a GradientDescentOptimizer.
  * This is the only optimizer which I have examined in any detail mathematically, I would like to experiment with others though. Changing from the ProximalAdagradOptimizer was a matter of necessity however.
3. Used mean squared error for my loss function.
  * I'm not sure what loss function the pre-built estimators use but this choice was made similarly to the optimizer
4. Shrunk the network to two hidden layers of 10 and 5 nodes respectively
  * It occurred to me that with only a 12 node input layer and a single node output layer, it makes very little intuitive sense to have hidden layers larger than 12.  This way, the network can be thought of as a funnel, concentrating the information present in the inputs into ultimately a single node.  This is not very mathematical but it did produce the best results in the least amount of time so I will definitely explore this idea further.
5. Used l2 regularization to avoid overfitting
  * The sensitivity and specificity on prediction weren't looking great so I added l2 regularization to the weights.

Let's try using this new model to predict other types of drug usage.


```python
feature_columns = [
    age_buckets, gender_buckets, education_buckets,
    nscore, ss
]

feature_col_names = process_column_names(feature_columns)

model_params = {
    "feature_columns": feature_columns,
    "hidden_layers": [10],
    "start_learn": 0.15,
    "end_learn": 0.01,
    "beta": 0.001
}

nn = tf.estimator.Estimator(model_fn, params=model_params)
train_and_eval(nn, "alcohol", feature_col_names)
custom_predict(nn, "alcohol", feature_col_names)
compare_results(["alcohol"])
```

    {'accuracy': 0.81, 'loss': 0.52362716, 'rmse': 0.94839674, 'global_step': 100000}
    Accuracy: 81.00%
    Comparing 200:200
    alcohol Accuracy: 81.000000%
    alcohol Sensitivity: 82.901554%
    alcohol Specificity: 28.571429%



```python
feature_columns = [
    gender_buckets,
    nscore, escore, cscore
]

feature_col_names = process_column_names(feature_columns)

model_params = {
    "feature_columns": feature_columns,
    "hidden_layers": [25,10,5],
    "start_learn": 0.1,
    "end_learn": 0.01,
    "beta": 0.001
}

nn = tf.estimator.Estimator(model_fn, params=model_params)
train_and_eval(nn, "nicotine", feature_col_names, steps=60000)
custom_predict(nn, "nicotine", feature_col_names)
compare_results(["nicotine"])
```

    {'accuracy': 0.61666667, 'loss': 0.70668542, 'rmse': 0.98140591, 'global_step': 60000}
    Accuracy: 61.67%
    Comparing 200:200
    nicotine Accuracy: 53.000000%
    nicotine Sensitivity: 51.041667%
    nicotine Specificity: 54.807692%


This was a bit surprising to me.  I tried many different combinations of hidden layers, feature columns, learning rate, and even the number of total training steps to see if I could get an accuracy above 70% and did not succeed.  The authors of the [paper](https://arxiv.org/abs/1506.06297) accompanying this dataset claim to have achieved around 70% sensitivity and specifity for each drug, so I think I will read over their findings and see if I can apply some of them to my own model.  Although if the authors were only able to achieve 70% on classifying nicotine usage, then I'm not too far off.


```python
feature_columns = [
    age_buckets, gender_buckets, country_buckets, ethnicity_buckets,
    education_buckets, tf.feature_column.embedding_column(nscore_ascore_cscore, 1),
    nscore, escore, oscore, ascore, cscore, impulsive, ss
]

feature_col_names = process_column_names(feature_columns)

model_params = {
    "feature_columns": feature_columns,
    "hidden_layers": [30,10],
    "start_learn": 0.15,
    "end_learn": 0.01,
    "beta": 0.001
}

nn = tf.estimator.Estimator(model_fn, params=model_params)
train_and_eval(nn, "LSD", feature_col_names)
custom_predict(nn, "LSD", feature_col_names)
compare_results(["LSD"])
```

    {'accuracy': 0.89999998, 'loss': 0.32547399, 'rmse': 0.82999963, 'global_step': 100000}
    Accuracy: 90.00%
    Comparing 200:200
    LSD Accuracy: 82.500000%
    LSD Sensitivity: 23.809524%
    LSD Specificity: 89.385475%


At first an accuracy over 90% looked awesome, but I was curious to see the sensitivity in particular.  Since LSD usage as defined occurs in such a small minority of the respondents, the model adopts a heavy negative bias.  It is very bad at identifying people who actually use LSD, and its accuracy is buoyed by the fact that this failure doesn't come up very often.

### Pruning Selection Features

After reading Table 18 on page 34 of [the study accompanying the dataset](https://arxiv.org/pdf/1506.06297.pdf)
I decided to be much more selective about which features to use.  They recommend at most 6 features and at
fewest 2 features should be used for a given classifier.


```python
feature_columns = [
    age_buckets, education_buckets, gender_buckets,
    nscore, ss
]

feature_col_names = process_column_names(feature_columns)

model_params = {
    "feature_columns": feature_columns,
    "hidden_layers": [10],
    "start_learn": 0.15,
    "end_learn": 0.01,
    "beta": 0.001
}

nn = tf.estimator.Estimator(model_fn, params=model_params)
train_and_eval(nn, "alcohol", feature_col_names)
custom_predict(nn, "alcohol", feature_col_names)
compare_results(["alcohol"])
```

    {'accuracy': 0.82666665, 'loss': 0.51349771, 'rmse': 0.93510514, 'global_step': 100000}
    Accuracy: 82.67%
    Comparing 200:200
    alcohol Accuracy: 82.000000%
    alcohol Sensitivity: 82.741117%
    alcohol Specificity: 33.333333%

