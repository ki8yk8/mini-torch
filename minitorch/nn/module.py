from collections import OrderedDict
from ..helpers import add_indent
from ..autograd import Value
import pickle as pkl

class Module:
	def __init__(self):
		self._parameters = OrderedDict()    # Value
		self._modules = OrderedDict()       # Linear, Neuron
		self.training = False

	def eval(self):
		self.training = False
		for modules in self._modules.values():
			modules.eval()

	def train(self):
		self.training = True
		for modules in self._modules.values():
			modules.train()

	def named_parameters(self):
		for name, param in self._parameters.items():
			yield name, param

		for name, module in self._modules.items():
			for subname, param in module.named_parameters():
				yield f"{name}.{subname}", param
	
	def parameters(self):
		for name, params in self.named_parameters():
			yield params

	def state_dict(self, weights_only=True):
		state_dict = OrderedDict()
		for name, params in self.named_parameters():
			state_dict[name] = params.data if weights_only else params

		return state_dict

	def load_state_dict(self, path):
		with open(path, "rb") as fp:
			state_dict = pkl.load(fp)

		for name, param in self.named_parameters():
			param.data = state_dict[name]
	
	def __setattr__(self, name, value):
		if isinstance(value, Value):
			self._parameters[name] = value
		elif isinstance(value, list):
			for i, p in enumerate(value):
				if isinstance(p, Value):
					self._parameters[f"{name}.{i}"] = p
				elif isinstance(p, Module):
					self._modules[f"{name}.{i}"] = p
		elif isinstance(value, Module):
			self._modules[name] = value
		
		super().__setattr__(name, value)

	def forward(self):
		pass

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def get_repr(self, child=0):
		inner_text = ""

		for name, module in self._modules.items():
			inner_text += add_indent(f"({name}): {module.get_repr(child+1)}\n", 2*(child+1))
		
		return f"{self.__class__.__name__}(\n{inner_text}"+add_indent(")", 2*child)

	def __repr__(self):
		return self.get_repr()
