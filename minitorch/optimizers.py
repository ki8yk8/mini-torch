class BaseOptimizer:
	def zero_grad(self):
		for params in self.parameters:
			params.grad = 0

class GD(BaseOptimizer):
	def __init__(self, parameters, lr=0.1):
		self.parameters = list(parameters)
		self.lr = lr

	def step(self):
		for params in self.parameters:
			params.data -= self.lr * params.grad
