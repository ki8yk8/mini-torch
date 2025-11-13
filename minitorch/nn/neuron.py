from .module import Module
from ..autograd import Value
from ..helpers import add_indent
import random

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

		return [z]
	
	def get_repr(self, child=0):
		return add_indent(
			f"Neuron(in_features={len(self.weights)}, bias={'True' if self.bias else 'False'})", 
			child
		)