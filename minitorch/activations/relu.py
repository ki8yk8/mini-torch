from ..nn import Module
from ..autograd import Value
from ..helpers import add_indent
from numbers import Number

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
		if isinstance(x, list):
			return [ReLU()(v) for v in x]
		elif isinstance(x, Number):
			x = Value(x)
		
		result = Value(data=relu(x), _child=(x,), _op="relu")

		def _backward():
			x.grad += result.grad * (0 if x.data <= 0 else 1)

		result._backward = _backward
		return result