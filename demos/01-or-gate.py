from minitorch.nn import Linear
from minitorch.optim import GD
import matplotlib.pyplot as plt

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 1, 1, 1]

LR, EPOCH = 0.1, 15

model = Linear(in_features=2, out_features=1, bias=True)
optimizer = GD(model.parameters(), lr=LR)

print(model)

losses = []
for i in range(EPOCH):
	total_loss = 0
	for x, actual in zip(inputs, outputs):
		predicted = model(x)[0]
		loss = (predicted-actual)**2
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		total_loss+=loss.data

	losses.append(total_loss)

plt.plot(range(EPOCH), losses)
plt.xlabel("Loss")
plt.ylabel("Epoch")
plt.grid()
plt.show()