import constants

def one_hot(num_classes, labels):
	result = [[0 for x in range(num_classes)] for z in labels]
	for i in range(len(labels)):
		result[i][labels[i]] = 1
	return result

def process_column_names(columns):
	return [process_column_name(x.name) for x in columns if process_column_name(x.name) is not None]

def process_column_name(name):
	if '_X_' in name:
		return None
	if '_bucketized' not in name:
		return name
	return name[:name.find('_bucketized')]

def decode(usage_code):
	# For use with 7 class targets
	# return str(constants.MAPPED_CODES[usage_code])

	if usage_code in constants.USER:
		return '1'
	else:
		return '0'

def attribute_column(attribute):
	return constants.NUMBERED_COLUMNS[attribute]

def compare_results(target):

	DATA_DIR="Dataset/"

	PREDICTIONS = DATA_DIR + "predictions.csv"
	TRUTH = DATA_DIR + "predict_drug_consumption_data.csv"

	HEADERS = True
	START = 1 if HEADERS else 0

	predictions = open(PREDICTIONS, "r")
	predictions_lines = predictions.readlines()
	truth = open(TRUTH, "r")
	truth_lines = truth.readlines()

	num_targets = len(target)

	print("Comparing " + str(len(predictions_lines) - START) + ":" + str(len(truth_lines) - START))

	total_correct = 0
	total_incorrect = 0

	correct_by_attr = {k: 0 for k in target}
	incorrect_by_attr = {k: 0 for k in target}

	correct_aggr = 0
	incorrect_aggr = 0

	true_positives = 0
	false_positives = 0
	true_negatives = 0
	false_negatives = 0

	if len(predictions_lines) == len(truth_lines):
		for x in range(START, len(predictions_lines)):
			all_correct = True
			for y in range(0, len(target)):

				attribute = target[y]
				prediction_line_split = predictions_lines[x].split(',')
				truth_line_split = truth_lines[x].split(',')

				if prediction_line_split[y].strip('\n') == decode(truth_line_split[attribute_column(attribute)]):
					if prediction_line_split[y].strip('\n') == '1':
						true_positives += 1
					else:
						true_negatives += 1
					total_correct += 1
					correct_by_attr[attribute] += 1
				else:
					if prediction_line_split[y].strip('\n') == '1':
						false_positives += 1
					else:
						false_negatives += 1
					all_correct = False
					total_incorrect += 1
					incorrect_by_attr[attribute] += 1

			if all_correct:
				correct_aggr += 1
			else:
				incorrect_aggr += 1

		# This is useless unless multiple targets are predicted.
		# print("Total Accuracy: %f" % (total_correct/(total_correct + total_incorrect)))
		for attr in target:
			print(attr + " Accuracy: %f%%" % 
				(correct_by_attr[attr]/(correct_by_attr[attr] + incorrect_by_attr[attr])*100))
			if (true_positives+false_positives) == 0:
				print(attr + " Sensitivity: N/A")
			else:
				print(attr + " Sensitivity: %f%%" %
					(true_positives/(true_positives+false_positives)*100))
			if (true_negatives+false_negatives) == 0:
				print(attr + " Specificity: N/A")
			else:
				print(attr + " Specificity: %f%%" %
					(true_negatives/(true_negatives+false_negatives)*100))

		# print("Aggregate Accuracy: %f" % (correct_aggr/(correct_aggr + incorrect_aggr)))

	else:
		print("Incorrect number of lines in " + PREDICTIONS)