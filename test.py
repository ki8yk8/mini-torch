from minitorch.losses import MSE

criteria = MSE()
print(criteria(1, 0))
print(criteria([1, 2], [1, 2]))
print(criteria([1, 2], [1, 3]))
print(criteria([1, 2], [3, 4]))
