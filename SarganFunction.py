import numpy as np
import scipy.optimize as spo

# Definimos la función objetivo corregida
def f(x):
    D = len(x)
    sum_square_terms = sum(x[i]**2 for i in range(D))  # Suma de los términos cuadráticos
    sum_cross_terms = 0.4 * sum(x[i] * x[j] for i in range(D) for j in range(D) if i != j)  # Términos cruzados
    return sum_square_terms + sum_cross_terms

# Definimos un punto inicial
x0 = [50, 20, -30, 40]  # Ajusta la dimensión D del problema aquí

# Utilizamos el método BFGS para minimizar la función
result = spo.minimize(f, x0, method='BFGS', tol=1.e-8, options={'maxiter': 100})

# Mostramos los resultados
print('Punto encontrado: ', result.x)
print('Valor mínimo de la función: ', result.fun)
print('Número de iteraciones: ', result.nit)
print('Resultado exitoso: ', result.success)
print('Mensaje del resultado: ', result.message)
