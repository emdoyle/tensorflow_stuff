old_file = open("drug_consumption_data.txt", "r")
output = open("re_drug_consumption_data.csv", "w")
test_output = open("test_drug_consumption_data.csv", "w")
predict_output = open("predict_drug_consumption_data.csv", "w")

DELIM = ","
NUM_ATTR = 31
NUM_TEST = 500
NUM_PREDICT = 100
HEADERS = False

old_lines = old_file.readlines()

if HEADERS:
	header = str(len(old_lines) - NUM_TEST - NUM_PREDICT) + DELIM + str(NUM_ATTR) + "\n"
	output.write(header)

	test_header = str(NUM_TEST) + DELIM + str(NUM_ATTR) + "\n"
	test_output.write(test_header)

	predict_header = str(NUM_PREDICT) + DELIM + str(NUM_ATTR) + "\n"
	predict_output.write(predict_header)

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