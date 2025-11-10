class BaseOptimizer:
	def zero_grad(self):
		for params in self.parameters:
			for p in params:
				p.grad = 0

class GD(BaseOptimizer):
	def __init__(self, parameters, lr=0.1):
		self.parameters = parameters
		self.lr = lr

	def step(self):
		for params in self.parameters:
			for p in params:
				p.data -= self.lr * p.grad
