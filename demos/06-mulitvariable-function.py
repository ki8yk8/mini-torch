from minitorch.nn import Module, Linear, Neuron
from minitorch.losses import MSE
from minitorch.optimizers import GD

import random

LR = 0.001
EPOCHS = 20

y = lambda x1, x2 : x1**2 + 2*x1*x2 + 3*x2**2

class MultiVariableDataset:
	def __init__(self, n):
		self.X = list(zip(
			[random.gauss(mu=0, sigma=1) for _ in range(n)], 
			[random.gauss(mu=1, sigma=-1) for _ in range(n)]
		))
		self.y = [y(x1, x2) for x1, x2 in self.X]

	def __len__(self):
		return len(self.y)

	def __getitem__(self, index):
		return self.X[index], self.y[index]

train_dataset = MultiVariableDataset(20)
test_dataset = MultiVariableDataset(10)

# creating model
class Model(Module):
	def __init__(self):
		super().__init__()
		self.linear = Linear(in_features=2, out_features=1)

	def forward(self, X):
		x = self.linear(X)

		return x[0]

model = Model()
print(model)

criterion = MSE()
optimizer = GD(model.parameters(), lr=LR)

def evaluate(model, criterion, dataset):
	total_loss = 0
	for X, y in dataset:
		output = model(X)
		total_loss += criterion(output, y)

	return total_loss.data/len(dataset)

print("\nTraining starts here====")
for i in range(EPOCHS):
	for x, y in train_dataset:
		output = model(x)
		loss = criterion(output, y)
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	train_loss = evaluate(model, criterion, train_dataset)
	test_loss = evaluate(model, criterion, test_dataset)

	print(f"Epoch {i+1}: train loss = {train_loss:.4f} test_loss = {test_loss:.4f}")
