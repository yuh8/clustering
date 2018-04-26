import pandas as pd
import numpy as np
df = pd.DataFrame(np.arange(12).reshape(4, 3))

# print(df.values)
# temp = np.arange(12).reshape(4, 3)
# print(2 * np.outer(np.ones(3), [1, 2, 3]))
# temp1 = temp - 8
# temp1[temp1 < 0] = 0
# print(temp1)
for x in df.values:
    print(x.shape)


def hello(x, *args):
    print(x)
    a, b = args[0:]
    a = a * 2
    b = b + 3
    return a, b


s = hello(0, np.array([1, 2, 3]), np.array([[4, 5, 6], [7, 8, 9]]))
print(s[0].T)
print(np.tile(s[0], (3, 1)).T)
