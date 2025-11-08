from minitorch.autograd import Value
from minitorch.activations import ReLU
import matplotlib.pyplot as plt

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]
lr = 0.1

w1, w2, b = Value(1.0), Value(-1.0), Value(2.0)

model = lambda x1, x2: w1 * x1 + w2 * x2 + b

total_loss = 0
for [x1, x2], actual in zip(inputs, outputs):
	predicted = model(x1, x2)
	loss = (predicted-actual)**2
	loss.backward()
	
	total_loss += loss.data
	print(w1.grad, w2.grad, b.grad)

	w1 -= lr * w1.grad
	w2 -= lr * w2.grad
	b -= lr * b.grad

	w1.grad = w2.grad = b.grad = 0

print(w1)
print(w2)
print(b)
print(total_loss)