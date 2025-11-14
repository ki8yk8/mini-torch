from ..module import Module
from ...math import exp
from ...helpers import add_indent

class Softmax(Module):
	def __init__(self, temperature=1.0):
		super().__init__()
		self.T = temperature

	def get_repr(self, child):
		return add_indent(f"Softmax(temperature={self.T})", child)

	def forward(self, x):
		if not isinstance(x, list):
			raise ValueError(f"Softmax only accepts type list, got {type(x)}")

		exponents = [exp(i/self.T) for i in x]
		probability = [e/sum(exponents) for e in exponents]
		
		return probability