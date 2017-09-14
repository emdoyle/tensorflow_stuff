import constants

PREDICTIONS = "predictions.csv"
TRUTH = "predict_drug_consumption_data.csv"

ATTRIBUTE_COLUMN = constants.NUMBERED_COLUMNS[constants.TARGET]
HEADERS = True
START = 1 if HEADERS else 0

predictions = open(PREDICTIONS, "r")
predictions_lines = predictions.readlines()
truth = open(TRUTH, "r")
truth_lines = truth.readlines()

print("Comparing " + str(len(predictions_lines)) + ":" + str(len(truth_lines)))
correct = 0
incorrect = 0

def decode(usage_code):
	if usage_code in constants.USER:
		return '1'
	else:
		return '0'

if len(predictions_lines) == len(truth_lines):
	# range begins with 1 to skip header
	for x in range(START, len(predictions_lines)):
		# annoying new lines are attached when read from the file
		# so I slice off last char
		if predictions_lines[x][:-1] == decode(truth_lines[x].split(',')[ATTRIBUTE_COLUMN]):
			correct += 1
		else:
			incorrect += 1

	print("Accuracy: %f" % (correct/(correct + incorrect)))

else:
	print("Incorrect number of lines in " + PREDICTIONS)