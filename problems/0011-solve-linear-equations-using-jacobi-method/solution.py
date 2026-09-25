import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:

	x = np.zeros_like(b, dtype=np.float64)

	for _ in range(n):
		x_next = np.zeros_like(x, dtype=np.float64)
		for i in range(len(b)):
			x_next[i] = (1/A[i][i]) * (b[i] - \
				np.sum([A[i][j] * x[j] for j in range(len(A[i])) if j != i]) )
		x = x_next

	return x