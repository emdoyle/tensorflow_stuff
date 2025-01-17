import os
import tensorflow as tf
import numpy as np
import pandas as pd
import constants

from tensorflow.contrib.learn.python.learn.datasets import base

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DRUG_TRAINING = "re_drug_consumption_data.csv"
DRUG_TEST = "test_drug_consumption_data.csv"
DRUG_PREDICT = "predict_drug_consumption_data.csv"

PREDICT_OUTPUT = "predictions.csv"

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

# Could bucketize but guessing these are close to linear
education = tf.feature_column.numeric_column("education")
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

base_columns = [
	age_buckets, gender_buckets, country_buckets, ethnicity_buckets,
	education, nscore, escore, oscore, ascore, cscore, impulsive,
	ss
]

crossed_columns = [
	nscore_ascore, nscore_cscore, ascore_cscore,
	nscore_ascore_cscore
]

feature_columns = base_columns + crossed_columns

classifier = tf.estimator.DNNLinearCombinedClassifier(
	model_dir=constants.MODEL_DIR,
	n_classes=2,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=base_columns,
    dnn_hidden_units=[200, 100, 50, 30, 10])

# classifier = tf.estimator.DNNClassifier(
# 	feature_columns=base_columns,
# 	n_classes=7,
# 	hidden_units=[1024, 512, 256],
#     optimizer=tf.train.ProximalAdagradOptimizer(
#       learning_rate=0.1,
#       l1_regularization_strength=0.001
#     ),
# 	model_dir=constants.MODEL_DIR)

# classifier = tf.estimator.LinearClassifier(
# 	feature_columns=feature_columns,
# 	n_classes=7,
# 	model_dir=constants.MODEL_DIR)

def input_fn(data_file, num_epochs, shuffle):
	dataset = pd.read_csv(
		tf.gfile.Open(data_file),
		header=0,
		usecols=constants.FEATURE_COLUMNS + constants.TARGETS,
		skipinitialspace=True,
		engine="python")
	# Drop NaN entries
	dataset.dropna(how="any", axis=0)

	# Init empty dataframe, add column for each of targets
	labels = pd.DataFrame(columns=constants.TARGETS)
	for target in constants.TARGETS:
		# Upper version of labels defines 7 classes (harder, lower accuracy)
		# labels[target] = dataset[target].apply(lambda x: constants.MAPPED_CODES[x]).astype(int)

		# Lower version of labels defines 2 classes (easier, higher accuracy)
		labels[target] = dataset[target].apply(lambda x: x in constants.USER).astype(int)

	return tf.estimator.inputs.pandas_input_fn(
		x=dataset,
		y=labels,
		batch_size=100,
		num_epochs=num_epochs,
		shuffle=shuffle,
		num_threads=1)

classifier.train(input_fn=input_fn(DRUG_TRAINING, num_epochs=None, shuffle=True),
	steps=1000)

results = classifier.evaluate(input_fn=input_fn(DRUG_TEST, num_epochs=1,
	shuffle=False),
	steps=1000)
# Only printing accuracy for now
# for key in sorted(results):
# 	print("%s: %s" % (key, results[key]))
print("Accuracy: %s" % results['accuracy'])

predictions = classifier.predict(input_fn=input_fn(DRUG_PREDICT, num_epochs=1,
	shuffle=False))
predict_writer = open(PREDICT_OUTPUT, "w")
predict_writer.write("Fake header\n")
for prediction in list(predictions):
	curr_line = ""
	for class_id in prediction['class_ids']:
		curr_line += (str(class_id) + ',')
	predict_writer.write(curr_line[:-1] + '\n')

predict_writer.close()