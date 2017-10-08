
# Using TensorFlow to Predict Drug Usage

### 1. DNNLinearCombinedClassifier


```python
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import constants

from tensorflow.contrib.learn.python.learn.datasets import base

# Less Verbose Output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

DATA_DIR = "Dataset/"
DRUG_TRAINING = DATA_DIR + "re_drug_consumption_data.csv"
DRUG_TEST = DATA_DIR + "test_drug_consumption_data.csv"
DRUG_PREDICT = DATA_DIR + "predict_drug_consumption_data.csv"

PREDICT_OUTPUT = DATA_DIR + "predictions.csv"

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
def input_fn(data_file, target, num_epochs, batch_size=30, shuffle=False, num_threads=1):
    dataset = pd.read_csv(
        tf.gfile.Open(data_file),
        header=0,
        usecols=constants.FEATURE_COLUMNS + [target],
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

    return tf.estimator.inputs.pandas_input_fn(
        x=dataset,
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1)
```


```python
base_columns = [
    age_buckets, gender_buckets, country_buckets, ethnicity_buckets,
    education_buckets,
    # For alcohol, purposefully removing personality features to reduce noise
    # nscore, escore,
    # oscore, ascore, cscore, impulsive,
    # ss
]

crossed_columns = [
    # See comment above
    # nscore_ascore, nscore_cscore, ascore_cscore,
    # nscore_ascore_cscore
]

feature_columns = base_columns + crossed_columns

classifier = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=constants.MODEL_DIR,
    n_classes=2,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=base_columns,
    dnn_hidden_units=[36],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.002,
        l2_regularization_strength=0.01))

classifier.train(input_fn=input_fn(DRUG_TRAINING, target="alcohol", num_epochs=None, shuffle=True),
    steps=60000)

results = classifier.evaluate(input_fn=input_fn(DRUG_TEST, target="alcohol", num_epochs=1,
    shuffle=False), steps=None)

print("Accuracy: %s" % results['accuracy'])

def predict(classifier, target):
    predictions = classifier.predict(input_fn=input_fn(DRUG_PREDICT, target=target, num_epochs=1,
        shuffle=False))
    predict_writer = open(PREDICT_OUTPUT, "w")
    predict_writer.write("Fake header\n")
    for prediction in list(predictions):
        curr_line = ""
        for class_id in prediction['class_ids']:
            curr_line += (str(class_id) + ',')
        predict_writer.write(curr_line[:-1] + '\n')

    predict_writer.close()

predict(classifier, "alcohol")
```

    Accuracy: 0.823333


Well, it's nowhere near the accuracy I saw with MNIST, but I think it's respectable.  I also separated out 200 cases on which I used `classifier.predict()` instead of `evaluate()`.  The cell below calculates accuracy, sensitivity and specifity on this sample of 200 using a separate [python script](https://github.com/emdoyle/tensorflow_stuff/tree/master/RealLearning/compare_results.py).


```python
from compare_results import compare_results
compare_results(["alcohol"])
```

    Comparing 200:200
    alcohol Accuracy: 83.000000%
    alcohol Sensitivity: 83.589744%
    alcohol Specificity: 60.000000%


It's worth noting (if only for a chance at interpretation) that the specifity in this sample is far below the sensitivity. This means that it was much more difficult for the model to predict that someone _didn't_ drink alcohol than if they did.  I suspect that this is because the model assumes a very positive bias, since guessing positively is usually correct.  Later it may be possible to correct this, but for now I will move on to other usages, since the optimal hyperparameters are likely to be different for different targets.


```python
base_columns = [
    age_buckets, gender_buckets, country_buckets, ethnicity_buckets,
    education_buckets,
]

# Although the individual personality traits aren't crossed columns,
# I think they will do better as input to the linear part of the model
crossed_columns = [
    nscore_ascore_cscore, nscore, escore, oscore, ascore, cscore, impulsive,
    ss
]

feature_columns = base_columns + crossed_columns

classifier = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=constants.MODEL_DIR + "_cannabis",
    n_classes=2,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=base_columns,
    dnn_hidden_units=[144, 72, 36, 18],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.002,
        l2_regularization_strength=0.005))

classifier.train(input_fn=input_fn(DRUG_TRAINING, target="cannabis", num_epochs=None, shuffle=True),
    steps=60000)

results = classifier.evaluate(input_fn=input_fn(DRUG_TEST, target="cannabis", num_epochs=1,
    shuffle=False), steps=None)

print("Accuracy: %s" % results['accuracy'])
```

    Accuracy: 0.85



```python
predict(classifier, "cannabis")
compare_results(["cannabis"])
```

    Comparing 200:200
    cannabis Accuracy: 76.000000%
    cannabis Sensitivity: 74.561404%
    cannabis Specificity: 77.906977%


I'm very surprised at the disparity between the `evaluate()` reported accuracy and the `predict()` reported accuracy.  It is possible that since the 200 is a small portion (~10%) of the total samples that it contains a significant number of outliers in terms of cannabis usage, or perhaps the model is overfitted to the data it saw.  I believe if the problem is overfitting that the solution is higher regularization strength, but if it is poor data then I will need to tweak my `constants` file to up the portion of the data used for prediction.


```python
classifier = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=constants.MODEL_DIR + "_cannabis",
    n_classes=2,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=base_columns,
    dnn_hidden_units=[72, 36, 18],
    dnn_optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.05,
        l2_regularization_strength=0.15))

classifier.train(input_fn=input_fn(DRUG_TRAINING, target="cannabis", num_epochs=None, shuffle=True),
    steps=60000)

results = classifier.evaluate(input_fn=input_fn(DRUG_TEST, target="cannabis", num_epochs=1,
    shuffle=False), steps=None)

print("Accuracy: %s" % results['accuracy'])
```

    Accuracy: 0.86



```python
predict(classifier, "cannabis")
compare_results(["cannabis"])
```

    Comparing 200:200
    cannabis Accuracy: 79.500000%
    cannabis Sensitivity: 78.899083%
    cannabis Specificity: 80.219780%


The regularization _does_ seem to have helped with the prediction metrics without much of an impact on evaluation accuracy.  Also I did remove the first hidden layer since I felt four hidden layers might be too much for a relatively simple binary classification.

### 2. Homemade Estimator

While the `DNNLinearCombinedClassifier` certainly performed well, to really have full control over the training model I will need to build an Estimator using `tf.estimator`.  Since I won't be using a pre-built classifier, I will need to create my own model function, which involves defining the layers of the network, loss, the optimizer, the learning rate, and whatever other parameters I choose to modify.  Below is the skeleton of a model function, taken from [here](https://www.tensorflow.org/extend/estimators).


```python
# def model_fn(features, labels, mode, params):
    # Logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # 2. Define the loss function for training/evaluation
    # 3. Define the training operation/optimizer
    # 4. Generate predictions
    # 5. Return predictions/loss/train_op/eval_metric_ops in EstimatorSpec object
#   return EstimatorSpec(mode, predictions, loss, train_op, eval_metric_ops)
```


```python
def input_fn(data_file, target, num_epochs, batch_size=30, shuffle=False, num_threads=1):
    dataset = pd.read_csv(
        tf.gfile.Open(data_file),
        header=0,
        usecols=constants.FEATURE_COLUMNS + [target],
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
        y=np.array(labels[target]),
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=1)
```


```python
def model_fn(features, labels, mode, params):
    
    # 1. Configure the model via TensorFlow operations
    input_layer = tf.cast(features["x"], tf.float32)
    
    layer_sizes = params["hidden_layers"]
    current_tensor = input_layer
    for nodes in layer_sizes:
        current_tensor = tf.layers.dense(current_tensor, nodes, activation=tf.nn.sigmoid)
        
    output_layer = tf.layers.dense(current_tensor, 1)

    # 4. Generate predictions
    predictions = tf.reshape(output_layer, [-1])
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"usage": predictions})
    
    # 2. Define the loss function for training/evaluation

    loss = tf.losses.mean_squared_error(labels, predictions)

    thresh_predictions = tf.where(tf.less(predictions, tf.constant(0.5, tf.float32)),
                             tf.zeros(tf.shape(predictions)), tf.ones(tf.shape(predictions)))
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(tf.cast(labels,tf.float64), tf.cast(predictions,tf.float64)),
        "accuracy": tf.metrics.accuracy(
            tf.cast(labels, tf.float64), tf.cast(thresh_predictions, tf.float64))
    }
    
    # 3. Define the training operation/optimizer
    decay_steps = 50000
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
base_columns = [
    age_buckets, gender_buckets, country_buckets, ethnicity_buckets,
    education_buckets,
]

# Although the individual personality traits aren't crossed columns,
# I think they will do better as input to the linear part of the model
# tf.feature_column.embedding_column(nscore_ascore_cscore, 1), 
crossed_columns = [
    nscore, escore, oscore, ascore, cscore, impulsive,
    ss
]

feature_columns = base_columns + crossed_columns

model_params = {
    "feature_columns": feature_columns,
    "hidden_layers": [30, 10],
    "start_learn": 0.1,
    "end_learn": 0.01
}

nn = tf.estimator.Estimator(model_fn, params=model_params)
```


```python
nn.train(input_fn=input_fn(DRUG_TRAINING, target="cannabis", batch_size=1, num_epochs=None, shuffle=True),
    steps=60000)

results = nn.evaluate(input_fn=input_fn(DRUG_TEST, target="cannabis", batch_size=1, num_epochs=1,
    shuffle=False), steps=None)

print(results)
print("Accuracy: %s" % results['accuracy'])
```

    {'accuracy': 0.85000002, 'loss': 0.10550805, 'rmse': 0.32482004, 'global_step': 60000}
    Accuracy: 0.85



```python
predictions = nn.predict(input_fn=input_fn(DRUG_PREDICT, target="cannabis", num_epochs=1,
    shuffle=False))
print(list(predictions))
```

    [{'usage': 0.59685236}, {'usage': 0.47341937}, {'usage': 0.64227188}, {'usage': 0.1248942}, {'usage': 0.14217383}, {'usage': -0.0045989156}, {'usage': 0.10255757}, {'usage': 0.25428391}, {'usage': 0.76051342}, {'usage': 0.11165461}, {'usage': 0.090698302}, {'usage': 0.60205126}, {'usage': 0.89982712}, {'usage': 0.60184252}, {'usage': -0.045811772}, {'usage': 0.84638214}, {'usage': 0.88041186}, {'usage': 0.21632466}, {'usage': 0.19042128}, {'usage': 0.15804395}, {'usage': -0.014212489}, {'usage': 0.010499775}, {'usage': 0.42756814}, {'usage': 0.28337499}, {'usage': 0.033432901}, {'usage': -0.075144827}, {'usage': 0.0045158863}, {'usage': 0.16458881}, {'usage': 0.038015842}, {'usage': 0.18464774}, {'usage': 0.10600266}, {'usage': 0.29981881}, {'usage': 0.00050014257}, {'usage': -0.04778403}, {'usage': 0.17212132}, {'usage': 0.017303109}, {'usage': 0.0479877}, {'usage': 0.10033041}, {'usage': 0.19028759}, {'usage': 0.33369297}, {'usage': 0.064602047}, {'usage': 0.78188229}, {'usage': 0.35418111}, {'usage': 0.11974186}, {'usage': 0.02665031}, {'usage': -0.0096550584}, {'usage': 0.025840878}, {'usage': -0.08590728}, {'usage': 0.54885453}, {'usage': 0.24312091}, {'usage': 0.16372219}, {'usage': 0.06198132}, {'usage': -0.016753554}, {'usage': 0.43495652}, {'usage': 0.56623816}, {'usage': -0.0030090213}, {'usage': 0.027764618}, {'usage': 0.18646187}, {'usage': 0.0071260333}, {'usage': 0.20330229}, {'usage': 0.82683462}, {'usage': 0.56678468}, {'usage': 0.22071192}, {'usage': 0.59268874}, {'usage': 0.10350475}, {'usage': 0.15815288}, {'usage': 0.89846784}, {'usage': 0.83941287}, {'usage': 0.88154632}, {'usage': -0.020616293}, {'usage': 0.36539292}, {'usage': 0.30774704}, {'usage': 0.082446545}, {'usage': 0.13579047}, {'usage': 0.80565995}, {'usage': 0.70547444}, {'usage': 0.60909021}, {'usage': 0.67208523}, {'usage': 0.58890694}, {'usage': 0.74066275}, {'usage': 0.58743149}, {'usage': 0.054942191}, {'usage': 0.77080798}, {'usage': 0.41865516}, {'usage': 1.0026687}, {'usage': 0.50997394}, {'usage': 0.72375816}, {'usage': 0.91082364}, {'usage': 0.72128773}, {'usage': 0.29146558}, {'usage': 0.60381943}, {'usage': 0.61909497}, {'usage': 0.3971923}, {'usage': 0.72362173}, {'usage': 0.43353567}, {'usage': 0.37378746}, {'usage': 0.80038428}, {'usage': 0.031379163}, {'usage': 1.0760726}, {'usage': 0.83495927}, {'usage': 0.30847806}, {'usage': 0.88695902}, {'usage': 0.92598057}, {'usage': 0.31462514}, {'usage': 0.77197635}, {'usage': 0.8960104}, {'usage': 0.77002317}, {'usage': 0.73046774}, {'usage': 0.10500827}, {'usage': 0.87598848}, {'usage': 0.80133998}, {'usage': -0.051026523}, {'usage': 0.62741232}, {'usage': 0.7629118}, {'usage': 0.9824692}, {'usage': 1.0199158}, {'usage': 0.92796755}, {'usage': 0.87196028}, {'usage': 0.38958108}, {'usage': 0.70707202}, {'usage': 0.051242888}, {'usage': 0.80938047}, {'usage': 0.82965022}, {'usage': 0.81195587}, {'usage': 0.67065334}, {'usage': 0.72961092}, {'usage': 0.5166775}, {'usage': 0.018323779}, {'usage': 0.33234823}, {'usage': 0.74621952}, {'usage': 0.58420223}, {'usage': 0.95600539}, {'usage': 0.88487339}, {'usage': 0.47237304}, {'usage': 0.8694014}, {'usage': 0.75905758}, {'usage': 0.4051066}, {'usage': 0.70528251}, {'usage': 0.72568506}, {'usage': 0.60485893}, {'usage': 0.52914673}, {'usage': 0.39195883}, {'usage': 0.48655069}, {'usage': 0.89105588}, {'usage': 0.88757461}, {'usage': 0.45175743}, {'usage': 0.75169796}, {'usage': 0.57835478}, {'usage': 0.52662039}, {'usage': 0.58995438}, {'usage': 0.37465668}, {'usage': 0.71245122}, {'usage': 0.40041858}, {'usage': 0.87896764}, {'usage': 0.57581425}, {'usage': 0.054318875}, {'usage': 0.95833015}, {'usage': 0.516137}, {'usage': 1.0074339}, {'usage': 0.74895364}, {'usage': 0.65006715}, {'usage': 0.41670719}, {'usage': 0.56036162}, {'usage': 0.9407528}, {'usage': 0.63555396}, {'usage': 0.11007315}, {'usage': 0.63103408}, {'usage': 0.90784717}, {'usage': 0.91378301}, {'usage': 0.89714611}, {'usage': 0.67590165}, {'usage': 0.79222155}, {'usage': 0.87993515}, {'usage': 0.94246906}, {'usage': 0.66921252}, {'usage': 0.44716117}, {'usage': 0.63426113}, {'usage': 0.62244016}, {'usage': 0.80120361}, {'usage': 0.76408756}, {'usage': 0.799474}, {'usage': 0.66347212}, {'usage': 0.67073244}, {'usage': 0.85601884}, {'usage': 0.98046803}, {'usage': 0.62996614}, {'usage': 0.82864445}, {'usage': 0.86226821}, {'usage': 0.85412771}, {'usage': 0.54651606}, {'usage': 0.87025779}, {'usage': 0.74291164}, {'usage': 0.79919815}, {'usage': 0.62132597}, {'usage': 0.94195247}, {'usage': 0.88782597}, {'usage': 1.0177544}, {'usage': 0.33364651}, {'usage': 0.79698002}, {'usage': 1.0391749}]
