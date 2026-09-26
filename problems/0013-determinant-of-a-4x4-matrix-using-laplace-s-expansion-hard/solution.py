import numpy as np
def determinant_4x4(matrix: list[list[int|float]]) -> float:
	
	def determinant(matrix: list[list[int|float]]) -> float:\

		matrix = np.array(matrix)
		
		if len(matrix) < 2:
			raise Exception(f"Matrix too small {len(matrix)}")

		if len(matrix) == 2:
			return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

		minor_determinants = np.array([])

		for i in range(len(matrix)):
			index_set = [j for j in range(len(matrix)) if j!= i]
			minor = matrix[1:,index_set]

			cofactor = 1 if i % 2 == 1 else -1

			minor_determinants = np.append(minor_determinants, determinant(minor) * matrix[0][i] * cofactor)

		return np.sum(minor_determinants)

	return determinant(matrix)
