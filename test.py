from minitorch.autograd import Value
from minitorch.nn import Neuron, Linear, Module
from minitorch.activations import ReLU, Sigmoid
from minitorch.optimizers import GD
import matplotlib.pyplot as plt

model = Neuron(in_features=2, bias=True)
print(model.forward([1, 2]))
# print(model.parameters())