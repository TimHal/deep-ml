import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	flat_list = np.array(a).flatten()

	if new_shape[0] * new_shape[1] != len(flat_list):
		return []

	res = []

	for i in range(new_shape[0]):
		offset = new_shape[1]
		res.append(flat_list[i*offset:(i*offset)+new_shape[1]])

	return res