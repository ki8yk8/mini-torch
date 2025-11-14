from minitorch import Value
from minitorch.math import log, sin

# Reference paper: https://www.csie.ntu.edu.tw/~cjlin/papers/autodiff/autodiff.pdf
# Forward Ref
x1 = Value(2)
x2 = Value(5)
v1 = log(x1)
v2 = x1*x2
v3 = sin(x2)
v4 = v1+v2 
v5 = v4-v3

print("\nForward Mode:")
print(f"\tx1 = {x1}")
print(f"\tx2 = {x2}")
print(f"\tv1 = {v1}")
print(f"\tv2 = {v2}")
print(f"\tv3 = {v3}")
print(f"\tv4 = {v4}")
print(f"\tv5 = {v5}")

# Reverse Mode
v5.backward()

print("\nReverse Mode:")
print(f"\tx1 = {x1}")
print(f"\tx2 = {x2}")
print(f"\tv1 = {v1}")
print(f"\tv2 = {v2}")
print(f"\tv3 = {v3}")
print(f"\tv4 = {v4}")
print(f"\tv5 = {v5}")

# you can verify the answers from the papers too