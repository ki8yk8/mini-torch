from ..nn import Module
from ..autograd import Value
from ..helpers import add_indent
from numbers import Number
import math

class Tanh(Module):
	def __init__(self):
		super().__init__()
	
	def get_repr(self, child=0):
		return add_indent("Tanh()", child)

	def forward(self, x):
		if isinstance(x, list):
			return [Tanh()(e) for e in x]
		elif isinstance(x, Number):
			x = Value(x)
		elif isinstance(x, Value):
			pass
		else:
			raise ValueError(f"Tanh expects list, number, Value, got {type(x)}")

		t = math.tanh(x.data)
		result = Value(t, _child=(x,), _op="tanh")

		def _backward():
			return 1-t*t
		result._backward = _backward

		return result