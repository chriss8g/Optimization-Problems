import random
import numpy as np
import numpy.linalg as npl
import scipy.optimize as spo
import matplotlib.pyplot as plt

def f(x):
    return sum(i * (xi**4) for i, xi in enumerate(x, 1)) + random.uniform(0, 1)

def df(x):
    return np.array([4 * i * (xi**3) for i, xi in enumerate(x, 1)])

def g(lambda_k, x, r):
    return f(x - lambda_k * r)

def steepestdescent(f, df, x0, tol=1.e-8, maxit=50):
    xk = x0
    x = [xk]
    r = df(xk)
    iters = 0
    
    while (npl.norm(r) > tol and iters < maxit):
        # Minimizar f(x - lambda_k * r) para encontrar lambda_k óptimo
        lambda_k = spo.golden(g, args=(xk, r))
        
        # Actualizar xk y r
        xk = xk - lambda_k * r
        r = df(xk)
        x.append(xk)
        
        iters += 1
    
    return x

x0 = np.array([1, -1, 0.5, 0.3, -0.4, 0.8, 1.4])
trajectory = steepestdescent(f, df, x0, tol=1.e-8, maxit=10)
print('x = ', trajectory[-1])

# Visualización
xmesh, ymesh = np.mgrid[-1.28:1.28:50j,-1.28:1.28:50j]
fmesh = np.array([[f([xmesh[i,j], ymesh[i,j]]) for j in range(xmesh.shape[1])] for i in range(xmesh.shape[0])])
plt.axis("equal")
plt.contour(xmesh, ymesh, fmesh, 20)
trajectory_array = np.array(trajectory)
plt.plot(trajectory_array.T[0], trajectory_array.T[1], "x-")
plt.show()
