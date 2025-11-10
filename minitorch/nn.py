from .autograd import Value

import random

class Module:
	def __init__(self):
		self._parameters = {}    # Value
		self._modules = {}    # Linear, Neuron

	def named_parameters(self):
		pass
	
	def parameters(self):
		for param in self._parameters.values():
			yield param

		for child in self._modules.values():
			yield param
	
	def __setattr__(self, name, value):
		if isinstance(value, Value):
			self._parameters[name] = value
		elif isinstance(value, list):
			self._parameters[name] = value
		
		super().__setattr__(name, value)

	def forward(self):
		pass

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

class Neuron(Module):
	def __init__(self, in_features, bias=True):
		super().__init__()
		self.weights = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
		self.bias = Value(random.uniform(-1, 1)) if bias else None

	def forward(self, x):
		if len(x) != len(self.weights):
			raise Exception(f"initialized in_features ({len(self.weights)}) not equal to provided one ({len(x)})")
		
		z = sum([self.weights[i]*x[i] for i in range(len(x))])
		if self.bias:
			z += self.bias

		return z
	
	def parameters(self):
		if (self.bias): 
			return [*self.weights, self.bias]

		return [*self.weights]
	
	def __repr__(self):
		return f"Neuron(in_features={len(self.weights)}, bias={'True' if self.bias else 'False'})"
	
class Linear(Module):
	def __init__(self, in_features, out_features, bias=True):
		super().__init__()
		self.neurons = [Neuron(in_features=in_features, bias=bias) for _ in range(out_features)]

	def forward(self, x):
		return [neuron(x) for neuron in self.neurons]

	def parameters(self):
		params = []

		for n in self.neurons:
			params = [*params, *n.parameters()]
		return params
	
	def __repr__(self):
		return f"Linear(in_features={len(self.neurons[0].weights)}, out_features={len(self.neurons)}, bias={'True' if self.neurons[0].bias else 'False'})"