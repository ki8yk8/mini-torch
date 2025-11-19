from minitorch.nn import Module, Neuron
from minitorch.nn import MSE
import matplotlib
import PyQt6
matplotlib.use("qtagg")

import matplotlib.pyplot as plt
import pickle

LR = 0.001
EPOCHS = 500

y = lambda x : 2*x+3
train_x, test_x = [x for x in range(0, 20, 2)], [x for x in range(5, 15, 3)]
train_y, test_y = [y(x) for x in train_x], [y(x) for x in test_x]

print(f"Create simple dataset (y=2*x+3) with train = {len(train_x)} and test = {len(test_x)}")

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

def evaluate(model, criterion, X, Y):
	outputs = [model([x]) for x in X]
	loss = criterion(outputs, Y)

	return loss.data/len(X)

prev_loss = evaluate(model, criterion, test_x, test_y)
print(f"Before loading, the model loss = {prev_loss}")
# saving the model
# model.save("demos/assets/simple-poly.pth")

# loading the model
model.load_state_dict("demos/assets/simple-poly.pth")

after_loss = evaluate(model, criterion, test_x, test_y)
print(f"After loading, from previous checkpoint loss = {after_loss}")

