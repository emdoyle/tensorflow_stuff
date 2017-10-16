#!/bin/bash

DOCS_DIR="/Users/evanmdoyle/Programming/Learn/docs/"
SRC_DIR="/Users/evanmdoyle/Downloads/"
SCRIPT_DIR="/Users/evanmdoyle/Programming/Learn/scripts/"
declare -a NOTEBOOK_NAMES=("VisualizingData" "RealLearning" "TensorFlowDeepNN" "DeepNNClassifier" "NEAT")

for name in ${NOTEBOOK_NAMES[@]}; do
	if [[ -e ${SRC_DIR}${name}.md || -e ${SRC_DIR}${name}.zip ]]
	then
		mv ${SRC_DIR}${name}* ${DOCS_DIR}
		if [[ -e ${DOCS_DIR}${name}.zip ]]
		then
			unzip ${DOCS_DIR}${name}.zip -d ${DOCS_DIR}
			mv ${DOCS_DIR}*.png ${DOCS_DIR}assets/
			rm ${DOCS_DIR}*.zip
		fi
		python ${SCRIPT_DIR}add_layout.py ${DOCS_DIR}${name}.md
	fi
done

/bin/bash ${SCRIPT_DIR}refresh_images.sh