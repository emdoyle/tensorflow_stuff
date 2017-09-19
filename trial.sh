#!/bin/bash

for var in "$@"
do
	if [ "$var" = "-r" ]
	then
		rm -r /tmp/drug_model;
	fi

	if [ "$var" = "-d" ]
	then
		python reformat_data.py;
	fi
done

echo "Train and evaluate:"
python estimators.py;

echo "Compare results:"
python compare_results.py;

echo "Number of 1s in predictions.csv:"
grep -o "1" predictions.csv | wc -l

echo "Number of 0s in predictions.csv:"
grep -o "0" predictions.csv | wc -l