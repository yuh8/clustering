import pandas as pd
import numpy as np

s = np.array([1, 2, 3, 4])
n = np.zeros((8))
l = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(np.array(l).shape)
x = np.linspace(1, 10, 10)
y = np.linspace(1, 10, 10)
X, Y = np.meshgrid(x, y)
print(X[0:10, :])
print(Y[0:10, :])
print(Y.shape)


# Test a simpler model
