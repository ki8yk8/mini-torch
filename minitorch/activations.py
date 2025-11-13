from .autograd import Value
from .math import exp, log
from .nn import Module
from .helpers import add_indent
import math
import numbers

def sanitize(x):
	sanitized = None

	if isinstance(x, list):
		sanitized = x
	elif isinstance(x, numbers.Number):
		sanitized = Value(x)
	elif isinstance(x, Value):
		sanitized = x
	else:
		raise("Unknown datatype received in activation")

	return sanitized

def sigmoid(x):
	if isinstance(x, Value):
		x = x.data

	return 1/(1+math.exp(-x))

def relu(x):
	if isinstance(x, Value):
		x = x.data

	return max(0, x)

class ReLU(Module):
	def __init__(self):
		super().__init__()

	def get_repr(self, child):
		return add_indent("ReLU()", child)
	
	def forward(self, x):
		x = sanitize(x)

		if isinstance(x, list):
			return [ReLU()(v) for v in x]
		
		result = Value(data=relu(x), _child=(x,), _op="relu")

		def _backward():
			x.grad += result.grad * (0 if x.data <= 0 else 1)

		result._backward = _backward
		return result

class Sigmoid(Module):
	def __init__(self):
		super().__init__()

	def get_repr(self, child=0):
		return add_indent("Sigmoid()", child)
	
	def forward(self, x):
		x = self.sanitize(x)

		if isinstance(x, list):
			return [Sigmoid()(v) for v in x]
		
		if x.data >= 0:
			return 1/(1+exp(-x))
		else:
			return exp(-x)/(1+exp(-x))

class Softmax(Module):
	def __init__(self, temperature=1.0):
		super().__init__()
		self.T = temperature

	def get_repr(self, child):
		return add_indent(f"Softmax(temperature={self.T})", child)

	def forward(self, x):
		exponents = [exp(i/self.T) for i in x]
		probability = [e/sum(exponents) for e in exponents]
		
		return probability

class LogSoftmax(Module):
	def __init__(self, stability=True):
		super().__init__()
		self.stability = stability

	def get_repr(self, child):
		return add_indent(f"LogSoftmax(stability={self.stability})", child)

	def forward(self, X):
		max_x = max(X) if self.stability else 0
		exponents = [exp(x-max_x) for x in X]

		return [x - max_x - log(sum(exponents)) for x in X]
