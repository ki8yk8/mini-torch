from ..module import Module
from ...helpers import add_indent
from ...math import exp, log

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