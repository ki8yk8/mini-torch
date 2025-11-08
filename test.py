from minitorch.autograd import Value
from minitorch.activations import ReLU
import matplotlib.pyplot as plt

# AND Gate training
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor = [0, 0, 0, 1]

lr = 2
EPOCHS = 1

w1, w2, b = Value(200), Value(300), Value(10)
history = {
	"w1": [w1],
	"w2": [w2],
	"b": [b],
}

model = lambda x1, x2: ReLU(w1 * x1 + w2 * x2 + b)
print("\nInitially")
print(f"\tx1 = 1, x2 = 0 => {model(1, 0).data}")

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

	history["b"].append(b)
	history["w1"].append(w1)
	history["w2"].append(w2)

print("\nFinally:")
print(f"\tx1 = 0, x2 = 0 => {model(0, 0).data}")
print(f"\tx1 = 0, x2 = 1 => {model(0, 1).data}")
print(f"\tx1 = 1, x2 = 0 => {model(1, 0).data}")
print(f"\tx1 = 1, x2 = 1 => {model(1, 1).data}")

plt.subplot(2, 2)
plt.plot(range(EPOCHS+1), [w.data for w in history["w2"]])
plt.grid()
plt.show()