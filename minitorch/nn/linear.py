from .module import Module
from .neuron import Neuron
from ..helpers import add_indent

class Linear(Module):
	def __init__(self, in_features, out_features, bias=True):
		super().__init__()
		self.neurons = [Neuron(in_features=in_features, bias=bias) for _ in range(out_features)]

	def forward(self, x):
		return [neuron(x)[0] for neuron in self.neurons]

	def get_repr(self, child=0):
		return add_indent(
			f"Linear(in_features={len(self.neurons[0].weights)}, out_features={len(self.neurons)}, bias={'True' if self.neurons[0].bias else 'False'})",
			child
		)