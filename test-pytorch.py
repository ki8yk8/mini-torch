import torch
from torch.nn import Module, Linear

class Model(Module):
	def __init__(self):
		super().__init__()
		self.linear1 = Linear(in_features=2, out_features=2, bias=True)
		self.linear2 = Linear(in_features=2, out_features=1, bias=True)

	def forward(self, x):
		linear1 = self.linear1(x)
		linear2 = self.linear2(linear1)
		base = torch.Tensor(2, requires_grad=True)

		return linear2
	
model = Model()
print(model)