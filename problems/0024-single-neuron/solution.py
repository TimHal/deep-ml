import math

def sigmoid(x):
	return 1/(1 + math.exp(-x))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):

	# Your code here
	probabilities, errors = [], []

	for i in range(len(features)):
		w_sum = sum([f * w for f,w in zip(features[i], weights)])
		w_sum += bias 

		res = sigmoid(w_sum)

		probabilities.append(round(res, 4))
		errors.append((res - labels[i]) ** 2)

	mse = sum(errors) / len(errors)

	return probabilities, round(mse, 4)