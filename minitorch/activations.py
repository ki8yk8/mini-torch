from .autograd import Value
import math
import numbers

class Activation:
	def __init__(self):
		pass

	def sanitize(self, x):
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

class ReLU(Activation):
	def __init__(self):
		super().__init__()


	def __call__(self, x):
		x = self.sanitize(x)

		if isinstance(x, list):
			return [ReLU()(v) for v in x]
		
		result = Value(data=relu(x), _child=(x,), _op="relu")

		def _backward():
			x.grad += result.grad * (0 if x.data <= 0 else 1)

		result._backward = _backward
		return result

class Sigmoid(Activation):
	def __call__(self, x):
		x = self.sanitize(x)

		if isinstance(x, list):
			return [Sigmoid()(v) for v in x]
		
		result = Value(sigmoid(x.data), _child=(x,), _op="sigmoid")

		def _backward():
			x.grad += result.grad * sigmoid(x.data) * (1-sigmoid(x.data))

		result._backward = _backward
		return result