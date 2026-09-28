import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X, dtype=np.float32)
	y = np.array(y, dtype=np.float32)

	theta = np.linalg.inv(X.T @ X) @ X.T @ y 
	return theta