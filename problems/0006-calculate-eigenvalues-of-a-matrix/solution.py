import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	trace_A = matrix[0][0] + matrix[1][1]
	det_A = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

	l0 = trace_A/2 + math.sqrt((trace_A/2)**2 - det_A)
	l1 = trace_A/2 - math.sqrt((trace_A/2)**2 - det_A)

	res = [l0, l1]
	res.sort()

	return res[::-1]