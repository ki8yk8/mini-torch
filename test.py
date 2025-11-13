from minitorch.nn import Module, Linear, Sequential
from minitorch.activations import ReLU, Sigmoid, LogSoftmax
from minitorch.optim import GD

class Model(Module):
	def __init__(self):
		super().__init__()
		self.linear = Linear(2, 4)
		self.relu = ReLU()
		self.sigmoid = Sigmoid()
		self.logsoftmax = LogSoftmax(False)

	def forward(self, X):
		X = self.linear(X)
		X = self.relu(X)
	
		return X

model = Model()
print(model)
optimizer = GD(model.parameters(), lr=0.01)

model.train()
model.eval()
print(model.sigmoid.training)