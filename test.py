from minitorch.nn import Module, Linear, Neuron
from minitorch.losses import MSE
from minitorch.optimizers import GD

LR = 0.1
EPOCHS = 100

y = lambda x : 2*x+3
train_x, test_x = [x for x in range(0, 20, 2)], [x for x in range(5, 15, 3)]
train_y, test_y = [y(x) for x in train_x], [y(x) for x in test_x]

print(f"Create simple dataset with train = {len(train_x)} and test = {len(test_x)}")

# creating model
class Model(Module):
	def __init__(self):
		super().__init__()
		self.neuron = Neuron(in_features=1)

	def forward(self, x):
		x = self.neuron(x)

		return x[0]

model = Model()
print(model)

criterion = MSE()
optimizer = GD(model.parameters(), lr=LR)

for i in range(EPOCHS):
	total_loss = 0
	for x, y in zip(train_x, train_y):
		output = model([x])
		loss = criterion(output, y)
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss.data

	print(f"Epoch {i+1}: train loss = {total_loss/len(train_x)}")