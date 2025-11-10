from minitorch.nn import Module, Linear, Neuron
from minitorch.optimizers import GD
import matplotlib.pyplot as plt

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 1]

LR, EPOCH = 0.1, 15

class Model(Module):
	def __init__(self):
		super().__init__()
		self.linear1 = Linear(in_features=2, out_features=4, bias=True)
		self.linear2 = Linear(in_features=4, out_features=2, bias=True)
		self.neuron = Neuron(in_features=2, bias=True)

	def forward(self, x):
		op = self.linear1(x)
		op = self.linear2(op)
		op = self.neuron(op)

		return op

model = Model()
optimizer = GD(model.parameters(), lr=LR)

losses = []
for i in range(EPOCH):
	total_loss = 0
	for x, actual in zip(inputs, outputs):
		predicted = model(x)
		loss = (predicted-actual)**2
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		total_loss+=loss.data

	losses.append(total_loss)

plt.plot(range(EPOCH), losses)
plt.show()