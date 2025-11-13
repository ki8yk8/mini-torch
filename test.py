from minitorch.nn import Module, Linear, Sequential
from minitorch.activations import ReLU, Sigmoid, LogSoftmax

# class Model(Module):
# 	def __init__(self):
# 		super().__init__()
# 		self.linear = Linear(2, 4)
# 		self.relu = ReLU()
# 		self.sigmoid = Sigmoid()
# 		self.logsoftmax = LogSoftmax(False)

# 	def forward(self, X):
# 		X = self.linear(X)
# 		X = self.relu(X)
	
# 		return X

# model = Model()
# print(model)
# output = model([1, 2])
# output[0].backward()

model = Sequential(
	Linear(10, 12),
	Linear(10, 12)
)

print(model)