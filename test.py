"""Implements training on MNIST dataset using minitorch"""
import pandas as pd
from minitorch.autograd import Value
from minitorch.nn import Linear, Module
from minitorch.math import exp
from minitorch.activations import Sigmoid, Softmax

train_df = pd.read_csv("./dataset/mnist_train.csv", header=None)
test_df = pd.read_csv("./dataset/mnist_test.csv", header=None)

print(f"Loaded train dataset with size = {len(train_df)}")
print(f"Loaded test dataset with size = {len(test_df)}\n")

# getting the x, y for train and test
train_x, train_y = train_df.iloc[:, train_df.columns != 0].values, train_df[0].values
test_x, test_y = test_df.iloc[:, test_df.columns != 0].values, test_df[0].values

# normalizing the train and test x
def normalize(X):
	return [[float(x)/255 for x in item] for item in X]

train_x, test_x = normalize(train_x), normalize(test_x)

# hyperparameters
LR = 0.1
EPOCH = 10

# creating the model
class MNISTClassifier(Module):
	def __init__(self):
		super().__init__()
		self.hidden = Linear(in_features=28*28, out_features=512)
		self.output = Linear(in_features=512, out_features=10)
		self.sigmoid = Sigmoid()

	def forward(self, x):
		x = self.hidden(x)
		x = self.sigmoid(x)
		x = self.output(x)

		return x
	
model = MNISTClassifier()
print(model)

print(test_x[0])