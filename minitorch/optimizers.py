class BaseOptimizer:
	def zero_grad(self):
		for p in self.parameters:
			p.grad = 0

class GD(BaseOptimizer):
	def __init__(self, parameters, lr=0.1):
		self.parameters = parameters
		self.lr = lr

	def step(self):
		for p in self.parameters:
			p.data -= self.lr * p.grad
