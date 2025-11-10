from minitorch.nn import Module, Linear, Neuron
from minitorch.optimizers import GD
import matplotlib.pyplot as plt

class Model(Module):
	def __init__(self):
		super().__init__()
		self.linear1 = Linear(in_features=2, out_features=4, bias=True)
		self.linear2 = Linear(in_features=4, out_features=2, bias=True)
		self.neuron = Neuron(in_features=2, bias=True)

	def forward(self, x):
		op = self.linear1(x)
		op = self.linear2(x)
		op = self.neuron(x)

		return op

model = Model()

