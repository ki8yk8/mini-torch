from minitorch.nn import Module, Linear, Neuron
from minitorch.activations import ReLU
from minitorch.autograd import Value
from minitorch.optimizers import GD
import matplotlib.pyplot as plt

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

print(f"x1=0, x2=0 => {model([0, 0]).data > 0.5}")
print(f"x1=0, x2=1 => {model([0, 1]).data > 0.5}")
print(f"x1=1, x2=0 => {model([1, 0]).data > 0.5}")
print(f"x1=1, x2=1 => {model([1, 1]).data > 0.5}")