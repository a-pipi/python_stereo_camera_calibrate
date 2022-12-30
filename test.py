import numpy as np
from scipy.spatial.transform import Rotation

a = np.cross([1,0,0], [0,0,1])
b = np.cross([0,1,0], [0,0,1])

nu = np.cross(a,b)
s = np.linalg.norm(nu)
c = a*b
I = np.eye(3)

nu_x = np.array([[0       ,-nu[2] ,nu[1]],
                 [nu[2]   ,0      ,-nu[0]],
                 [-nu[1]  ,nu[0]  ,0]])

kaas = 1/(1+c)

R = I+nu_x+nu_x*nu_x*kaas
r = Rotation.from_matrix(R)

print(np.dot(a,R), b)