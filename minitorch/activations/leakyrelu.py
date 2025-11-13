from ..nn import Module
from ..autograd import Value
from ..helpers import add_indent
import numbers

class LeakyReLU(Module):
	def __init__(self, negative_slope=0.01):
		self.negative_slope = negative_slope

	def get_repr(self, child):
		return add_indent(f"LeakyReLU(negative_slope={self.negative_slope})", child)
	
	def forward(self, X):
		if isinstance(X, Value):
			X = [X]
		elif isinstance(X, numbers.Number):
			X = [Value(X)]
		else:
			raise ValueError(f"LeakyReLU only accepts type Value, or Number or list of either, got {type(X)}")
		
		out = []
		for x in X:
			if x > 0:
				out.append(x)
			else:
				out.append(self.negative_slope * x)
		
		return out if len(X) > 1 else out[0]