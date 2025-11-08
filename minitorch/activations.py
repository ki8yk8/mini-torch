from .autograd import Value
import math

def ReLU(x):
	relu = 0 if x.data <= 0 else x.data
	result = Value(data=relu, _child=(x), _op="relu")

	def _backward():
		x.grad += result.grad * 0 if x.data <= 0 else 1

	result._backward = _backward
	return result

def sigmoid(x):
	return 1/(1+math.exp(-x))

def Sigmoid(x):
	result = Value(sigmoid(x.data), _child=(x), _op="sigmoid")

	def _backward():
		x.grad += result.grad * sigmoid(x.data)/(1-sigmoid(x.data))

	result._backward = _backward
	return result