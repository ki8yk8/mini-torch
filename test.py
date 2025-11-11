from minitorch.autograd import Value
from minitorch.activations import ReLU
from minitorch.nn import Module, Linear, Neuron


# xor gate
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 0]

LR, EPOCH = 0.01, 4

class Model(Module):
	def __init__(self):
		super().__init__()
		self.linear1 = Linear(in_features=2, out_features=4, bias=True)
		self.linear2 = Linear(in_features=4, out_features=2, bias=True)
		self.neuron = Neuron(in_features=2, bias=True)
		self.bias = Value(0)

	def forward(self, x):
		op = self.linear1(x)
		op = ReLU(op)
		op = self.linear2(op)
		op = self.neuron(op)
		op += self.bias

		return op

class NewModel(Module):
	def __init__(self):
		super().__init__()
		self.linear1 = Linear(10, 5, bias=False)
		self.model1 = Model()
		self.model2 = Model()

model = NewModel()
print(model)