from ..module import Module

class MSE(Module):
	def __init__(self):
		super().__init__()

	def forward(self, output, actual):
		if not isinstance(output, list) or not isinstance(actual, list):
			raise ValueError(f"Both output and actual should be of type list")
		
		if len(output) != len(actual):
			raise ValueError(f"Mismatch size of output and actual, {len(output)} != {len(actual)}")
		
		squared_error = [(o-a)**2 for o, a in zip(output, actual)]
		
		return sum(squared_error)/len(squared_error)