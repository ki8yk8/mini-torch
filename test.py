from minitorch.autograd import Value
from minitorch.nn import Neuron, Linear, Module
from minitorch.activations import ReLU, Sigmoid
from minitorch.optimizers import GD
import matplotlib.pyplot as plt

class CustomLinear(Module):
	def __init__(self, in_features, out_features, bias=True):
		self.neurons = [Neuron(in_features=in_features, bias=bias) for _ in range(out_features)]

	def forward(self, x):
		return [neuron(x) for neuron in self.neurons]

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]
lr = 0.1
EPOCHS = 15

model = CustomLinear(in_features=2, out_features=1)
print(model.parameters())
raise
optimizer = GD(model.parameters(), lr=lr)


losses = []
for i in range(EPOCHS):
	total_loss = 0
	for x, actual in zip(inputs, outputs):
		predicted = model(x)[0]
		loss = (predicted-actual)**2
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss.data

	losses.append(total_loss)

print(model((0, 0))[0].data > 0.5)
print(model((0, 1))[0].data > 0.5)
print(model((1, 0))[0].data > 0.5)
print(model((1, 1))[0].data > 0.5)

plt.plot(range(EPOCHS), losses)
plt.show()