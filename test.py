from minitorch.autograd import Value
from minitorch.nn import Neuron, Linear, Module
from minitorch.optimizers import GD
import matplotlib.pyplot as plt

class Model(Module):
	def __init__(self):
		pass

	def forward(self, x, y, z):
		print(x, y, z)

model = Model()
model(1, 2, z=3)

# inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
# outputs = [0, 0, 0, 1]
# lr = 0.1
# EPOCHS = 15

# model = Linear(in_features=2, out_features=1)
# print(model)
# optimizer = GD(model.parameters(), lr=lr)

# losses = []
# for i in range(EPOCHS):
# 	total_loss = 0
# 	for x, actual in zip(inputs, outputs):
# 		predicted = model(x)[0]
# 		loss = (predicted-actual)**2
# 		loss.backward()

# 		optimizer.step()
# 		optimizer.zero_grad()

# 		total_loss += loss.data

# 	losses.append(total_loss)

# print(model((0, 0))[0].data > 0.5)
# print(model((0, 1))[0].data > 0.5)
# print(model((1, 0))[0].data > 0.5)
# print(model((1, 1))[0].data > 0.5)

# plt.plot(range(EPOCHS), losses)
# plt.show()