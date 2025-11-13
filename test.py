from minitorch.nn import RNN

model = RNN(input_size=12, hidden_size=64, num_layers=1)
print(model)