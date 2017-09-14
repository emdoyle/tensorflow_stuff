import constants
old_file = open("drug_consumption_data.txt", "r")
output = open("re_drug_consumption_data.csv", "w")
test_output = open("test_drug_consumption_data.csv", "w")
predict_output = open("predict_drug_consumption_data.csv", "w")

DELIM = ","
NUM_ATTR = 31
NUM_TEST = 500
NUM_PREDICT = 100
HEADERS = True

old_lines = old_file.readlines()

if HEADERS:
	attr_names = ""
	for attr in constants.CSV_COLUMNS:
		attr_names += attr
		attr_names += DELIM

	# Cut off last DELIM
	header = attr_names[:-1]

	output.write(header + '\n')
	test_output.write(header + '\n')
	predict_output.write(header + '\n')

for line in old_lines[:-(NUM_TEST + NUM_PREDICT)]:
	output.write(line[(line.find(",")+1):])

for line in old_lines[-(NUM_TEST + NUM_PREDICT):-(NUM_PREDICT)]:
	test_output.write(line[(line.find(",")+1):])

for line in old_lines[-(NUM_PREDICT):]:
	predict_output.write(line[(line.find(",")+1):])

old_file.close()
output.close()
test_output.close()
predict_output.close()