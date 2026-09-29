import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	normalized_data, standardized_data = np.copy(data), np.copy(data)
	
	mean, std = np.mean(data, axis=0), np.std(data, axis=0)
	standardized_data = (standardized_data - np.tile(mean, (len(data), 1)) ) / np.tile(std, (len(data), 1))

	min_cols, max_cols = np.min(data, axis=0), np.max(data, axis=0)
	scale = max_cols - min_cols

	min_cols = np.tile(min_cols, (len(data), 1))
	scale = np.tile(scale, (len(data), 1))

	normalized_data = (normalized_data - min_cols) / scale

	return standardized_data, normalized_data