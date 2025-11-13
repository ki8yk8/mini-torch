from minitorch.nn import Module, Linear, Sequential
from minitorch.activations import ReLU, Sigmoid, LogSoftmax
import minitorch

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

print(model.state_dict())

model.load_state_dict("./state_dict.pth")

print(model.state_dict())