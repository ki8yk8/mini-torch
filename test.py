from minitorch.nn import RNN
from minitorch.autograd import Value

model = RNN(input_size=4, hidden_size=4, num_layers=2)
print(model)

x = [
	[Value(1), Value(2), Value(3), Value(4)], 
	[Value(1), Value(2), Value(3), Value(4)], 
	[Value(1), Value(2), Value(3), Value(4)],
]
model(x)