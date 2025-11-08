import math

class Value:
	def __init__(self, data, _child=(), _op=""):
		self.data = data
		self._child = _child
		self._op = _op
		self._backward = lambda: None

		self.grad = 0    # this is by default constant

	def __mul__(self, other):
		result = Value(self.data * other.data, _child=(self, other), _op="*")
		
		def _backward():
			self.grad += result.grad * other.data    # use the multiplication rule if there is multiplication of two variables then only other variable remains
			other.grad += result.grad * self.data

		result._backward = _backward
		return result

	def __truediv__(self, other):
		result = Value(self.data/other.data if other.data != 0 else 0, _child=(self, other), _op="/")

		def _backward():
			if other.data == 0:
				self.grad += 0
				other.grad += 0
			else:
				self.grad += result.grad * other.data/other.data**2
				other.grad += result.grad * -self.data/other.data**2

		result._backward = _backward
		return result
	
	def __add__(self, other):
		result = Value(self.data + other.data, _child=(self, other), _op="+")

		def _backward():
			self.grad += 1 * result.grad    # use addition rule and find
			other.grad += 1 * result.grad

		result._backward = _backward

		return result
	
	def __sub__(self, other):
		result = Value(self.data - other.data, _child=(self, other), _op="-")

		def _backward():
			self.grad += result.grad * 1
			other.grad += result.grad * -1
		
		result._backward = _backward
		return result
		
	def cos(self):
		result = Value(math.cos(self), _child=(self), _op="cos")

		def _backward():
			self.grad += result.grad * -math.sin(self.data)
		
		result._backward = _backward
		return result
	
	def tan(self):
		result = Value(math.tan(self), _child=(self), _op="tan")

		def _backward():
			self.grad += result.grad * 1/(math.cos(self.data)**2)
		
		result._backward = _backward
		return result
	
	def exp(self):
		result = Value(math.exp(self.data), _child=(self), _op="e")

		def _backward():
			self.grad += result.grad * math.exp(self.data)
		
		result.exp = _backward
		return result

	def __repr__(self):
		return f"Value({self.data:.4f}, grad={self.grad:.4f})"
	
	def backward(self):
		self._backward()