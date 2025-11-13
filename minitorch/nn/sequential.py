from .module import Module
from ..helpers import add_indent

class Sequential(Module):
	def __init__(self, *layers):
		super().__init__()
		self.layers = layers

	def get_repr(self, child=0):
		representation = add_indent(f"Sequential(\n", child)

		for layer in self.layers:
			representation += add_indent(layer.get_repr(child+1))
			representation += "\n"

		representation += add_indent(")", child)
		return representation
		

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x