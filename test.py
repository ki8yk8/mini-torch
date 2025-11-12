"""Implements training on MNIST dataset using minitorch"""
import pandas as pd
from minitorch.autograd import Value
from minitorch.nn import Linear, Module
from minitorch.math import exp
from minitorch.activations import Sigmoid, Softmax

class MnistDataset:
	def __init__(self, csv_path, transform=None, target_transform=None):
		df = pd.read_csv(csv_path, header=None)
		self.X = df.iloc[:, df.columns !=0].values
		self.y = df[0].values
		self.transform = transform
		self.target_transfrom = target_transform

	def __len__(self):
		return len(self.y)
	
	def __getitem__(self, index):
		x, y = self.X[index], self.y[index]

		if self.transform:
			x = self.transform(x)

		if self.target_transfrom:
			y = self.target_transfrom(y)

		return x, y

def normalize_x(X):
	return [float(x) for x in X]

def normalize_y(y):
	return int(y)

train_set = MnistDataset("./dataset/mnist_train.csv", transform=normalize_x, target_transform=normalize_y)
test_set = MnistDataset("./dataset/mnist_test.csv", transform=normalize_x, target_transform=normalize_y)

print(f"Loaded dataset with train = {len(train_set)} and test = {len(test_set)}")

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

print(test_set[0])