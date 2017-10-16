#!/bin/bash

BASE_DIR="/Users/evanmdoyle/Programming/Learn/"
DOCS_DIR="${BASE_DIR}docs/"
ASSETS_DIR="${BASE_DIR}docs/assets/"
SCRIPT_DIR="${BASE_DIR}scripts/"
declare -a NOTEBOOK_NAMES=("VisualizingData" "RealLearning" "TensorFlowDeepNN" "DeepNNClassifier" "NEAT")

for name in ${NOTEBOOK_NAMES[@]}; do
	for f in ${BASE_DIR}${name}/*.png; do
		if [[ -e ${f} ]]
		then
			cp ${f} ${ASSETS_DIR}
		fi
	python ${SCRIPT_DIR}fix_image_paths.py ${DOCS_DIR}${name}.md
	done
done