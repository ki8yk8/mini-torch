from minitorch.autograd import Value
from minitorch.nn import Neuron
import matplotlib.pyplot as plt

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]
lr = 0.1
EPOCHS = 15

# w1, w2, b = Value(2.0), Value(-1.0), Value(2.0)

# model = lambda x1, x2: w1 * x1 + w2 * x2 + b
model = Neuron(in_features=2, bias=True)
print(model.parameters())

losses = []
for i in range(EPOCHS):
	total_loss = 0
	for x, actual in zip(inputs, outputs):
		predicted = model(x)
		loss = (predicted-actual)**2
		loss.backward()
		
		total_loss += loss.data

	losses.append(total_loss)

print(model((0, 0)).data > 0.5)
print(model((0, 1)).data > 0.5)
print(model((1, 0)).data > 0.5)
print(model((1, 1)).data > 0.5)

plt.plot(range(EPOCHS), losses)
plt.show()