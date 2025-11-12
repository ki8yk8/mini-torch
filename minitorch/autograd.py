class Value:
	def __init__(self, data, _child=(), _op=""):
		self.data = data
		self._child = set(_child)
		self._op = _op
		self._backward = lambda: None

		self.grad = 0.0    # this is by default constant

	def __mul__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		result = Value(self.data * other.data, _child=(self, other), _op="*")
		
		def _backward():
			self.grad += result.grad * other.data    # use the multiplication rule if there is multiplication of two variables then only other variable remains
			other.grad += result.grad * self.data

		result._backward = _backward
		return result

	def __truediv__(self, other):
		return self*other**-1
	
	def __add__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		result = Value(self.data + other.data, _child=(self, other), _op="+")

		def _backward():
			self.grad += 1 * result.grad    # use addition rule and find
			other.grad += 1 * result.grad

		result._backward = _backward

		return result
	
	def __sub__(self, other):
		other = other if isinstance(other, Value) else Value(other)
		result = Value(self.data - other.data, _child=(self, other), _op="-")

		def _backward():
			self.grad += result.grad * 1
			other.grad += result.grad * -1
		
		result._backward = _backward
		return result

	def __pow__(self, other):
		result = Value(self.data**other, (self,), _op=f"^{other}")

		def _backward():
			self.grad += result.grad * other * self.data**(other-1)
		
		result._backward = _backward
		return result
	
	def __neg__(self):
		return self*-1
	
	# means 1 + Value(10)
	def __radd__(self, other):
		return self + other
	
	# means 1 - Value(10)
	def __rsub__(self, other):
		return other - self
	
	def __rmul__(self, other):
		return self * other

	# means other/self
	def __rtruediv__(self, other):
		return self**-1 * other

	def __repr__(self):
		return f"Value({self.data:.4f}, grad={self.grad:.4f})"

	def __eq__(self, other):
		return self.data == other.data

	def __ne__(self, other):
		return self.data != other.data

	def __lt__(self, other):
		return self.data < other.data

	def __le__(self, other):
		return self.data <= other.data

	def __gt__(self, other):
		return self.data > other.data

	def __ge__(self, other):
		return self.data >= other.data
	
	def __hash__(self):
		# this is defined because when __eq__ was define it cleared __hash__ to avoid inconsistent hashing 
		return id(self)
	
	def backward(self):
		# topological ordering dfs
		visited = set()
		topology = []

		def build_topology(node):
			if node not in visited:
				visited.add(node)
				for child in node._child:
					build_topology(child)
				topology.append(node)
		
		build_topology(self)
		# seeding the gradient
		self.grad = 1.0
		for node in reversed(topology):
			node._backward()