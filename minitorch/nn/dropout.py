from .module import Module
from ..helpers import add_indent
from ..autograd import Value
import random

class Dropout(Module):
	def __init__(self, p=0.5):
		if p < 0.0 or p>1.0:
			raise ValueError(f"dropout probability has to be between 0.0 and 1.0, got {p}")
		self.p = p

	def get_repr(self, child):
		return add_indent(f"Dropout(p={self.p})", child)

	def forward(self, X):
		if self.training:
			if not isinstance(X, list):
				X = [X]
			
			out = []
			for x in X:
				if random.random() <= self.p:
					out.append(Value(0.0))
				else:
					out.append( x * (1.0 / (1.0 - self.p)))
		else:
			return X