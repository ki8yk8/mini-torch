from .autograd import Value

import random

class Module:
	def __init__(self):
		self._parameters = {}

	def parameters(self):
		return self._parameters
	
	def __setattr__(self, name, value):
		if isinstance(value, Value):
			self._parameters[name] = value
		elif isinstance(value, list):
			self._parameters[name] = value
		else:
			super().__setattr__(name, value)

	def forward(self):
		return None

	def __call__(self, *x, **y):
		return self.forward(*x, **y)

class Neuron(Module):
	def __init__(self, in_features, bias=True):
		super().__init__()
		self.weights = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
		self.bias = Value(random.uniform(-1, 1)) if bias else None

	def forward(self, x):
		if len(x) != len(self._parameters["weights"]):
			raise Exception(f"initialized in_features ({len(self.weights)}) not equal to provided one ({len(x)})")
		
		result = sum([self._parameters["weights"][i]*x[i] for i in range(len(x))])
		if "bias" in self._parameters:
			result += self._parameters["bias"]

		return result
	
	def __repr__(self):
		return f"Neuron(in_features={len(self.weights)}, bias={'True' if self.bias else 'False'})"
	
class Linear:
	def __init__(self, in_features, out_features, bias=True):
		self.neurons = [Neuron(in_features=in_features, bias=bias) for _ in range(out_features)]

	def __call__(self, x):
		return [neuron(x) for neuron in self.neurons]

	def parameters(self):
		params = []

		for n in self.neurons:
			params = [*params, *n.parameters()]
		return params
	
	def __repr__(self):
		return f"Linear(in_features={len(self.neurons[0].weights)}, out_features={len(self.neurons)}, bias={'True' if self.neurons[0].bias else 'False'})"