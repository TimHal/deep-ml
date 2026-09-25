import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

	def covar(A: list[float], B: list[float]) -> float:
		
		covar = 0
		A_mean, B_mean = sum(A) / len(A), sum(B) / len(B)

		for i in range(len(A)):
			covar += (A[i] - A_mean)*(B[i] - B_mean)

		return covar/(len(A) - 1)

	covar_matrix = np.empty((len(vectors), len(vectors)))

	for i in range(len(vectors)):
		for j in range(len(vectors)):
			covar_matrix[i][j] = covar(vectors[i], vectors[j])

	return covar_matrix