from sklearn.preprocessing import MinMaxScaler
import numpy as np

sc = MinMaxScaler()
sc_vals = [[-1, 0], [1, 2]]
sc.fit(sc_vals)

try_list = [[1,2], [0,3], [-1, 4]]
try_list = np.array(try_list)

b = [1,2]

print(b)

a = sc.transform(b)

print(a)