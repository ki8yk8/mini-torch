from ..module import Module
from ...activations import LogSoftmax
from .nll import NLL

class CrossEntropy(Module):
	def __init__(self):
		super().__init()
		self.log_softmax = LogSoftmax(stability=True)

	def forward(self, output, target):
		if target > len(output) -1:
			raise Exception(f"The target {target} is out of bound of output")

		log_probs = self.log_softmax(output)
		
		return NLL()(log_probs, target)

