from minitorch.nn import CrossEntropy
from minitorch import Value
import torch.nn as nn
import torch

def test_cross_entropy():
	torch_ce = nn.CrossEntropyLoss()
	ce = CrossEntropy()

	raw_logits, target = [1.0, 12.0, 14.0], 1

	mini_logits, mini_target = [Value(i) for i in raw_logits], target
	torch_logits, torch_target = torch.Tensor(raw_logits), torch.tensor(target)

	assert ce(mini_logits, mini_target).data == torch_ce(torch_logits, torch_target)