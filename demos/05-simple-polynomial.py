from minitorch.nn import Module, Neuron
from minitorch.nn import MSE
from minitorch.optim import GD
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
optimizer = GD(model.parameters(), lr=LR)

def evaluate(model, criterion, X, Y):
	outputs = [model([x]) for x in X]
	loss = criterion(outputs, Y)

	return loss.data/len(X)

train_losses, test_losses = [], []
print("\nTraining starts here====")
for i in range(EPOCHS):
	for x, y in zip(train_x, train_y):
		output = model([x])
		loss = criterion([output], [y])
		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	train_loss = evaluate(model, criterion, train_x, train_y)
	test_loss = evaluate(model, criterion, test_x, test_y)
	
	train_losses.append(train_loss)
	test_losses.append(test_loss)

	print(f"Epoch {i+1}: train loss = {train_loss:.4f} test_loss = {test_loss:.4f}")

print(f"\nModeled equation = {model.neuron.weights[0].data:.3f} * x + {model.neuron.bias.data:.3f}")

plt.plot(range(EPOCHS), train_losses, label="Train")
plt.plot(range(EPOCHS), test_losses, label="Test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Change of loss with epoch")
plt.legend()
plt.grid()
plt.show()

# saving the model
with open("demos/assets/simple-poly.pth", "wb") as fp:
	pickle.dump(model.state_dict(), fp)