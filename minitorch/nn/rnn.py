from .module import Module
from .linear import Linear
from ..activations import Tanh, ReLU, Sigmoid
from ..autograd import Value
import random

class RNN(Module):
	def __init__(self, input_size, hidden_size, num_layers, nonlinearity="tanh", bias=True):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		
		self.x2a = Linear(in_features=input_size, out_features=hidden_size, bias=bias)
		self.h2a = Linear(in_features=hidden_size, out_features=hidden_size, bias=bias)

		# creating activation
		match nonlinearity:
			case "tanh":
				self.activation = Tanh()
			case "relu":
				self.activation = ReLU()
			case "sigmoid":
				self.activation = Sigmoid()
			case _:
				raise ValueError(f"nonlinearity has to be tanh, relu, or sigmoid, got {nonlinearity}")


	def forward(self, input, hx=None):
		if not len(input) == self.input_size:
			raise ValueError(f"Input must be a list of length equal to input size, {len(input)} != {self.input}")

		if hx and not len(hx) == self.hidden_size:
			raise ValueError(f"Initial hidden state mst be a list of length equal to hidden size, {len(hx)} != {self.hidden_size}")
		
		if not hx:
			hx = [random.random() for _ in range(self.hidden_size)]

		all_hiddens = []
		for i in self.num_layers:
			hx = self.x2a(input) + self.h2a(hx)
			hx = self.activation(hx)
			all_hiddens.append(hx)

		return hx, all_hiddens


