from minitorch import Value
from minitorch.nn.activations import Sigmoid, Tanh, ReLU, LeakyReLU
import matplotlib
import PyQt6
matplotlib.use("qtagg")

import matplotlib.pyplot as plt

def linspace(start, end, targets):
	if not isinstance(targets, int):
		raise ValueError(f"3rd argument of linspace must be int, got {type(targets)}")
	op = start
	difference = (end-start)/targets
	while op >= start and op <= end:
		yield op
		op += difference

X = list(linspace(-8, 8, 100))


# result of different activation
sigmoid = Sigmoid()
tanh = Tanh()
relu = ReLU()
leaky_rely = LeakyReLU(negative_slope=0.1)   # exxagerating for better visualization

op_sigmoid = [x.data for x in sigmoid(X)]
op_tanh = [x.data for x in tanh(X)]
op_relu = [x.data for x in relu(X)]
op_leaky_relu = [x for x in leaky_rely(X)]

# visualizing the output in matplotlib plot
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(X, op_sigmoid)
axes[0, 0].set_title("Sigmoid")
axes[0, 0].grid(True)

axes[0, 1].plot(X, op_tanh)
axes[0, 1].set_title("Tanh")
axes[0, 1].grid(True)

axes[1, 0].plot(X, op_relu)
axes[1, 0].set_title("ReLU")
axes[1, 0].grid(True)

axes[1, 1].plot(X, op_leaky_relu)
axes[1, 1].set_title("Leaky ReLU")
axes[1, 1].grid(True)

fig.suptitle("Different activation function visualized")
plt.show()