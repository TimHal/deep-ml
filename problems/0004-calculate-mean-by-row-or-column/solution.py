def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

	def transpose(matrix: list[list[float]]) -> list[list[float]]:
		res = []
		for i in range(len(matrix[0])):
			transposed_row = []

			for row in matrix:
				transposed_row.append(row[i])
			
			res.append(transposed_row)

		return res
	
	if mode == "row":
		means = []
		for row in matrix:
			means.append(sum(row) / len(row))

	else:
		# transpose and call again
		# using the transpose solution from a few problems ago here
		return calculate_matrix_mean(transpose(matrix), "row")

	return means