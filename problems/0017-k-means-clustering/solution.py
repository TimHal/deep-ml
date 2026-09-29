import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:


	def cdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
		# Implementation of scipy's cdist (kinda)
		# The idea is that A, B are a_m x n, b_m x n dimensional
		# inputs with n features each. cdist will return the pairwise euclidean
		# distance for each pair (a,b)

		dist_matrix = np.zeros((len(A), len(B)))

		for i in range(len(A)):
			for j in range(len(B)):
				# assuming same dimensionality here
				dist_ij = 0
				for x_i in range(len(A[i])):
					dist_ij += (A[i][x_i] - B[j][x_i]) ** 2

				dist_matrix[i][j] = dist_ij
		
		return dist_matrix

	points = np.array(points)
	centroids = np.array(initial_centroids, dtype=np.float32)
	distances = cdist(points, centroids)
	assignment = np.argmin(distances, axis=1)  # [m x data, n x centroids] and we select
		# the minimal index along the vertical axis so assignments should be 
		# an [len(data),] shaped array with entries == index of closest centroid 

	
	for i in range(max_iterations):

		# update centroids
		for c_i in range(len(centroids)):
			# select points based on current assignment
			c_i_data = points[assignment == c_i]
			centroids[c_i] = np.mean(c_i_data, axis=0)

		# recalculate distances 
		distances = cdist(points, centroids)

		# update assignment and check if it changed (early stopping, if not)
		assignment_new = np.argmin(distances, axis=1)

		if (assignment_new != assignment).any():
			assignment = assignment_new
		else:
			break

	return [tuple(c) for c in centroids]