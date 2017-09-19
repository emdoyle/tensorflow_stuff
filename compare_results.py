import constants

PREDICTIONS = "predictions.csv"
TRUTH = "predict_drug_consumption_data.csv"

HEADERS = True
START = 1 if HEADERS else 0

predictions = open(PREDICTIONS, "r")
predictions_lines = predictions.readlines()
truth = open(TRUTH, "r")
truth_lines = truth.readlines()

num_targets = len(constants.TARGETS)

print("Comparing " + str(len(predictions_lines) - START) + ":" + str(len(truth_lines) - START))

total_correct = 0
total_incorrect = 0

correct_by_attr = {k: 0 for k in constants.TARGETS}
incorrect_by_attr = {k: 0 for k in constants.TARGETS}

correct_aggr = 0
incorrect_aggr = 0

def decode(usage_code):
	# For use with 7 class targets
	# return str(constants.MAPPED_CODES[usage_code])

	if usage_code in constants.USER:
		return '1'
	else:
		return '0'

def attribute_column(attribute):
	return constants.NUMBERED_COLUMNS[attribute]

if len(predictions_lines) == len(truth_lines):
	for x in range(START, len(predictions_lines)):
		all_correct = True
		for y in range(0, len(constants.TARGETS)):

			attribute = constants.TARGETS[y]
			prediction_line_split = predictions_lines[x].split(',')
			truth_line_split = truth_lines[x].split(',')

			if prediction_line_split[y].strip('\n') == decode(truth_line_split[attribute_column(attribute)]):
				total_correct += 1
				correct_by_attr[attribute] += 1
			else:
				all_correct = False
				total_incorrect += 1
				incorrect_by_attr[attribute] += 1

		if all_correct:
			correct_aggr += 1
		else:
			incorrect_aggr += 1

	# This is useless unless multiple targets are predicted.
	# print("Total Accuracy: %f" % (total_correct/(total_correct + total_incorrect)))
	for attr in constants.TARGETS:
		print(attr + " Accuracy: %f" % 
			(correct_by_attr[attr]/(correct_by_attr[attr] + incorrect_by_attr[attr])))

	# print("Aggregate Accuracy: %f" % (correct_aggr/(correct_aggr + incorrect_aggr)))

else:
	print("Incorrect number of lines in " + PREDICTIONS)