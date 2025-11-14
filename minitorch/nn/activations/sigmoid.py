from ..module import Module
from ...autograd import Value
from ...helpers import add_indent
from ...math import exp
from numbers import Number

class Sigmoid(Module):
	def __init__(self):
		super().__init__()

	def get_repr(self, child=0):
		return add_indent("Sigmoid()", child)
	
	def forward(self, x):
		if isinstance(x, list):
			return [Sigmoid()(v) for v in x]
		elif isinstance(x, Number):
			x = Value(x)
		
		if x.data >= 0:
			return 1/(1+exp(-x))
		else:
			return exp(x)/(1+exp(x))