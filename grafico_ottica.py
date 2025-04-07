import matplotlib.pyplot as plt
import numpy as np
import openpyxl

from module import *

#ATT: la funzione prende alpha in radianti e i in gradi, sorry
def formula_delta_teorico(alpha, n, i):
    delta = i - np.rad2deg(alpha) + np.rad2deg(np.arcsin(np.sin(alpha)*(n**2 -np.sin(np.deg2rad(i))**2)**0.5 - np.cos(alpha)*np.sin(np.deg2rad(i))))
    return delta

#path = "/Users/ginevramingione/Desktop/Misure di Delta.xlsx"

path = "./Misure di Delta.xlsx"

C1 = "F"
C2 = "C"
FR1 = 3
LR1 = 12

i = listify(path,FR1,LR1,C1)
delta = listify(path,FR1,LR1,C2)
delta_delta = 0.5
d = np.linspace(min(i),max(i),1000)

alpha = np.deg2rad(60.1)
n = 1.51

plt.scatter(i,delta)
plt.errorbar(i,delta,delta_delta, fmt= "none")


#plt.plot(d, d-60.1+ np.rad2deg(np.arcsin(np.sin(alpha)*(1.51**2 -np.sin(np.deg2rad(d))**2)**0.5 - np.cos(alpha)*np.sin(np.deg2rad(d)))))

plt.plot(d, formula_delta_teorico(alpha, n, d))
plt.show()







