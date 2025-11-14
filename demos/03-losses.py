from minitorch.nn import MSE, CrossEntropy
from minitorch.autograd import Value

mse = MSE()
ce = CrossEntropy()

# demo of mse
logits, targets = [10.0, 12.0, 3.0], [2.0, 12.0, 3.0]
print(f"One target is far, MSE = {mse(logits, targets):.3f}")

logits, targets = [10.0, 12.0, 3.0], [10.0, 12.0, 3.0]
print(f"All targets and logits are identical, MSE = {mse(logits, targets):.3f}\n")

# demo of cross entropy
logits, target = [Value(1.0), Value(12.0), Value(3.0)], 1
print(f"When logits of target is high, cross entropy = {ce(logits, target).data:.4f}")

logits, target = [Value(8.0), Value(11.0), Value(9.0)], 1
print(f"When all logits are similar, cross entropy = {ce(logits, target).data:.4f}")

logits, target = [Value(1.0), Value(0.0), Value(3.0)], 1
print(f"When logits of target is low, cross entropy = {ce(logits, target).data:.4f}")