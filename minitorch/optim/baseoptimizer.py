class BaseOptimizer:
	def zero_grad(self):
		for params in self.parameters:
			params.grad = 0