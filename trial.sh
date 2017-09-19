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