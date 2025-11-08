from .autograd import Value
import math

def log(x):
	result = Value(math.log(x.data), _child=(x,), _op="log")

	def _backward():
		x.grad += result.grad * 1/x.data

	result._backward = _backward

	return result

def sin(x):
	result = Value(math.sin(x.data), _child=(x,), _op="sin")

	def _backward():
		x.grad += result.grad * math.cos(x.data)
	result._backward = _backward

	return result

def cos(x):
	result = Value(math.cos(x.data), _child=(x,), _op="cos")

	def _backward():
		x.grad += result.grad * -math.sin(x.data)
	
	result._backward = _backward
	return result

def tan(x):
	result = Value(math.tan(x.data), _child=(x,), _op="tan")

	def _backward():
		x.grad += result.grad * 1/(math.cos(x.data)**2)
	
	result._backward = _backward
	return result

def exp(x):
	result = Value(math.exp(x.data), _child=(x,), _op="e")

	def _backward():
		x.grad += result.grad * math.exp(x.data)
	
	result.exp = _backward
	return result
