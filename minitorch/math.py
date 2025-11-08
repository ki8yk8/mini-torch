from .autograd import Value
import math

def log(x):
	result = Value(math.log(x.data), _child=(x), _op="log")

	def _backward():
		x.grad += result.grad * 1/x.data

	result._backward = _backward

	return result

def sin(x):
	result = Value(math.sin(x.data), _child=(x), _op="sin")

	def _backward():
		x.grad += result.grad * math.cos(x.data)
	result._backward = _backward

	return result