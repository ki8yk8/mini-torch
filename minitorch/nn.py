from .autograd import Value
import random

class Neuron:
	def __init__(self, in_features, bias=True):
		self.weights = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
		self.bias = Value(random.uniform(-1, 1)) if bias else None

	def __call__(self, x):
		if len(x) != len(self.weights):
			raise Exception(f"initialized in_features ({len(self.weights)}) not equal to provided one ({len(x)})")
		
		result = sum([self.weights[i]*x[i] for i in range(len(x))])
		if self.bias:
			result += self.bias

		return result

	def parameters(self):
		params = [*self.weights]
		if self.bias:
			params.append(self.bias)

		return params