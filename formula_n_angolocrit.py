import numpy as np
from math import *

#angolo di prima incidenza critico
i0 = 28
delta_i0 = 0.5

#angolo al vertice
alpha = 60
delta_alpha=0.7


def n(i0, alpha0):
    i = radians(i0)
    alpha = radians(alpha0)
    A = (1 + sin(i)*cos(alpha))/(sin(alpha))
    return sqrt(A**2 + (sin(i))**2)

N = 1000
l = np.linspace(i0-delta_i0, i0 + delta_i0, N)
l2 = np.linspace(alpha - delta_alpha, alpha + delta_alpha, N)

A = []

for t in l:
    for a in l2:
        A.append(n(t,a))


print("valor 'medio': ", n(i0,alpha))

print("valore massimo: ",max(A))
print("valore minimi: ",min(A))
print( "incertezza da assegnare: ",0.5* (max(A)- min(A)))


