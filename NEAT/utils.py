import constants

def one_hot(num_classes, labels):
	result = [[0 for x in range(num_classes)] for z in labels]
	for i in range(len(labels)):
		result[i][labels[i]] = 1
	return result

def index_from_boundaries(feature, boundaries):
	boundaries.append(feature)
	temp = sorted(boundaries)
	return temp.index(feature)

def expand_features(cases):
	new_cases = []
	for features, usages in cases:
		long_features = []
		for i in range(len(features)):
			if i < len(constants.BOUNDARIES):
				boundaries = constants.BOUNDARIES[i]
				result = [0 for x in range(len(boundaries)+1)]
				if boundaries:
					result[index_from_boundaries(features[i],boundaries)] = 1
					long_features += result
				else:
					long_features.append(features[i])
			else:
				long_features.append(features[i])

		new_cases.append((long_features, usages))
	return new_cases

def list_diff(first, second):
    return [item for item in first if item not in second]
    
DRUG_INDEXES = {k:v for v, k in enumerate(list_diff(constants.CSV_COLUMNS, constants.FEATURE_COLUMNS))}