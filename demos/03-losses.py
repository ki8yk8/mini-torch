from minitorch.optim import MSE
from minitorch.autograd import Value

criterion = MSE()
loss = criterion([Value(1), Value(2)], [Value(3), Value(4)])
loss.backward()