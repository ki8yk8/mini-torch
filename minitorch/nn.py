import random
from collections import OrderedDict
import pickle as pkl

from .autograd import Value
from .helpers import add_indent

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

class Neuron(Module):
	def __init__(self, in_features, bias=True):
		super().__init__()
		self.weights = [Value(random.uniform(-1, 1)) for _ in range(in_features)]
		self.bias = Value(random.uniform(-1, 1)) if bias else None

	def forward(self, x):
		if len(x) != len(self.weights):
			raise Exception(f"initialized in_features ({len(self.weights)}) not equal to provided one ({len(x)})")
		
		z = sum([self.weights[i]*x[i] for i in range(len(x))])
		if self.bias:
			z += self.bias

		return [z]
	
	def get_repr(self, child=0):
		return add_indent(
			f"Neuron(in_features={len(self.weights)}, bias={'True' if self.bias else 'False'})", 
			child
		)
	
class Linear(Module):
	def __init__(self, in_features, out_features, bias=True):
		super().__init__()
		self.neurons = [Neuron(in_features=in_features, bias=bias) for _ in range(out_features)]

	def forward(self, x):
		return [neuron(x)[0] for neuron in self.neurons]

	def get_repr(self, child=0):
		return add_indent(
			f"Linear(in_features={len(self.neurons[0].weights)}, out_features={len(self.neurons)}, bias={'True' if self.neurons[0].bias else 'False'})",
			child
		)

class Sequential(Module):
	def __init__(self, *layers):
		super().__init__()
		self.layers = layers

	def get_repr(self, child=0):
		representation = add_indent(f"Sequential(\n", child)

		for layer in self.layers:
			representation += add_indent(layer.get_repr(child+1))
			representation += "\n"

		representation += add_indent(")", child)
		return representation
		

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)

		return x

class Dropout(Module):
	def __init__(self, p=0.5):
		if p < 0.0 or p>1.0:
			raise ValueError(f"dropout probability has to be between 0.0 and 1.0, got {p}")
		self.p = p

	def get_repr(self, child):
		return add_indent(f"Dropout(p={self.p})", child)

	def forward(self, X):
		if self.training:
			if not isinstance(X, list):
				X = [X]
			
			out = []
			for x in X:
				if random.random() <= self.p:
					out.append(Value(0.0))
				else:
					out.append( x * (1.0 / (1.0 - self.p)))
		else:
			return X