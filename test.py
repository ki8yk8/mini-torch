from minitorch.autograd import Value
from minitorch.activations import ReLU

# XOR Gate training
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
xor = [0, 1, 1, 0]
lr = 0.01

w1, w2, b = Value(200), Value(3000), Value(10)

model = lambda x1, x2: ReLU(w1 * x1 + w2 * x2 + b)
print(f"x1 = 1, x2 = 0 => {model(1, 0).data}")

for [x1, x2], actual in zip(data, xor):
	output = model(x1, x2)
	loss = (output - actual)**2
	loss.backward()

	# changing the weights and biases with respsect to those gradients
	w1 = w1 - lr * w1.grad
	w2 = w2 - lr * w2.grad
	b = b - lr * b.grad

print(f"x1 = 1, x2 = 0 => {model(1, 0).data}")