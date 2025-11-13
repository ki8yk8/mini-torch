from ..module import Module

# difference between NLL and CE is that NLL assumes the input is logsoftmax
class NLL(Module):
	def __init__(self):
		super().__init__()

	def forward(self, output, target):
		if not isinstance(output, list) or not isinstance(target, list):
			raise ValueError(f"Both output and actual should be of type list")

		if target > len(output) - 1:
			raise Exception(f"The target {target} is out of bound")

		return -output[target]