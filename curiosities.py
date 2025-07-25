from sklearn.preprocessing import MinMaxScaler
import numpy as np

sc = MinMaxScaler()
sc_vals = [[-1], [1]]
sc.fit(sc_vals)
""" 
try_list = [[1,2], [0,3], [-1, 4]]
try_list = np.array(try_list)
 """
b = sc.transform([[0]])
print(b)

a = [1,2]
a.extend([3,4])

print(a)

#a = sc.transform(b)

#print(a)