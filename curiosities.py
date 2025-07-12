from sklearn.preprocessing import MinMaxScaler
import numpy as np

sc = MinMaxScaler()
sc_vals = [[-1, 0], [1, 2]]
sc.fit(sc_vals)

try_list = [[1,2], [0,3], [-1, 4]]
try_list = np.array(try_list)


c = [1,2]
c = c.reshape(1,1)
a = np.array([[[4,5,6], [7,8,9]]])
b = np.array([[1,2,3]])
b = b.reshape(1,1,-1)
b = np.append(a, b, axis=1)

print(b)

#a = sc.transform(b)

#print(a)