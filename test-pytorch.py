import torch

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
outputs = [0, 0, 0, 1]
lr = 0.1

w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(-1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)

model = lambda x1, x2: w1 * x1 + w2 * x2 + b

total_loss = 0
for [x1, x2], actual in zip(inputs, outputs):
	predicted = model(x1, x2)
	loss = (predicted-actual)**2
	loss.backward()
	
	total_loss += loss.item()
	print(w1.grad, w2.grad, b.grad)

	with torch.no_grad():
		w1 -= lr * w1.grad
		w2 -= lr * w2.grad
		b -= lr * b.grad
	
	w1.grad = w2.grad = b.grad = None
	
# print(w1, w1.grad)
# print(w2, w2.grad)
# print(b, b.grad)
# print(total_loss)