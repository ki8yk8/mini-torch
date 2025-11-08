from minitorch.autograd import Value
from minitorch.activations import ReLU
import matplotlib.pyplot as plt

# AND Gate training
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor = [0, 0, 0, 1]

lr = 1
EPOCHS = 40

w1, w2, b = Value(0), Value(-1), Value(-1)
# w1, w2, b = Value(1), Value(1), Value(-1)
losses = []

model = lambda x1, x2: w1 * x1 + w2 * x2 + b
print("\nInitially")
print(f"\tx1 = 0, x2 = 0 => {model(0, 0).data}")
print(f"\tx1 = 0, x2 = 1 => {model(0, 1).data}")
print(f"\tx1 = 1, x2 = 0 => {model(1, 0).data}")
print(f"\tx1 = 1, x2 = 1 => {model(1, 1).data}")

for i in range(EPOCHS):
	for [x1, x2], actual in zip(data, xor):
		output = model(x1, x2)
		loss = (output - actual)**2
		loss.backward()

		# changing the weights and biases with respsect to those gradients
		w1 -= lr * w1.grad
		w2 -= lr * w2.grad
		b -= lr * b.grad

		# resetting the gradient
		w1.grad = 0
		w2.grad = 0
		b.grad = 0
	
	losses.append(loss.data)

print("\nFinally:")
print(f"\tx1 = 0, x2 = 0 => {model(0, 0).data}")
print(f"\tx1 = 0, x2 = 1 => {model(0, 1).data}")
print(f"\tx1 = 1, x2 = 0 => {model(1, 0).data}")
print(f"\tx1 = 1, x2 = 1 => {model(1, 1).data}")

plt.plot(range(EPOCHS), losses)
plt.title("Change of loss with epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()