from .baseoptimizer import BaseOptimizer

class GD(BaseOptimizer):
	def __init__(self, parameters, lr=0.1):
		self.parameters = list(parameters)
		self.lr = lr

	def step(self):
		for params in self.parameters:
			params.data -= self.lr * params.grad