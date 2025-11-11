import torch

a, b = torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
c = torch.sigmoid(a)
d = b*c
e = a+d

c.retain_grad()
d.retain_grad()
e.retain_grad()

e.backward()
print(a.item(), b.item(), c.item(), d.item(), e.item())
print(a.grad, b.grad, c.grad, d.grad, e.grad)

# 1.0 2.0 0.7310585975646973 1.4621171951293945 2.4621171951293945
# tensor(1.3932) tensor(0.7311)