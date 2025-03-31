import matplotlib.pyplot as plt
import numpy as np
import openpyxl

### SUITE DI STRUMENTI PYTHON SVILUPPATI PER LAB2
"""
Comprende funzioni per importare file da excel(listify),
fare fit lineari, non lineari, pesati etc

Utilizzo: mettere il file nella stessa cartella del .py e importarlo come un normale modulo python
"""



#Traduce lettere in indice di colonna Excel

def translate(col):
    supp = []
    supp1 = 0
    count1 = 0
    alpha = "abcdefghijklmnopqrstuvwxyz"
    alpha1 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in col:
        if i in alpha:
            count = 0
            for j in alpha:
                count += 1
                if i == j:
                    supp.append(count)
                    break
                else:
                    pass
                
        if i in alpha1:
            count = 0
            for j in alpha1:
                count += 1
                if i == j:
                    supp.append(count)
                    break
                else:
                    pass

    supp = supp[::-1]
    for i in supp:
        supp1 = supp1 + (25**count1)*int(i)
        count1 += 1
        
    return supp1


#Trasforma colonna excel in array

def listify(path, first_row, last_row, column):
    wb = openpyxl.load_workbook(path,data_only=True) 
    sheet_obj = wb.active
    supp1=[]
    

    FR = first_row
    LR = last_row
    FC = translate(column)
    

    for i in range(LR-FR+1):
        cell_obj = sheet_obj.cell(row = FR+i, column = FC)
        if(cell_obj.value == None):
            continue
        supp1.append(float(cell_obj.value))
        
    return np.array(supp1)


#Fit lineare non pesato

def linear_fit(x,y,sigmay):
    
    N = len(x)
    x_sq = 0
    y_i = 0
    x_i = 0
    xy = 0
    
    for i in range(N):
        x_sq = x_sq+ x[i]**2
        y_i = y_i + y[i]
        x_i = x_i + x[i]
        xy = xy + x[i]*y[i]
        
    delta = N*x_sq -(x_i**2)
    B = (1/delta)*((x_sq*y_i)-(x_i*xy))
    A = (1/delta)*((N*xy)-(x_i*y_i))
    sigmaB = ((sigmay**2)*x_sq/delta)**0.5
    sigmA = ((sigmay**2)*N/delta)**0.5
    covAB = (sigmay**2)*(-x_i)/(N*x_sq-(x_i)**2)

    return [A,B,sigmA,sigmaB,covAB]

#Fit di diretta proporzionalità non pesata

def dir_prop_fit(x,y,sigmay):
    
    N = len(x)
    x_sq = 0
    xy = 0
    for i in range(N):
        x_sq = x_sq+ x[i]**2
        xy = xy + x[i]*y[i]
    
    M = xy/x_sq
    sigmaM = ((sigmay**2)/x_sq)**0.5


    return [M,sigmaM]


#Fit Lineare Pesato

def w_linear_fit(x,y,sigmay):
    
    N = len(x)
    x_sq = 0
    y_i = 0
    x_i = 0
    xy = 0
    sig0 = 0
    
    for i in range(N):
        x_sq = x_sq+ (x[i]**2)*(1/sigmay[i]**2)
        y_i = y_i + y[i]*(1/sigmay[i]**2)
        x_i = x_i + x[i]*(1/sigmay[i]**2)
        xy = xy + x[i]*y[i]*(1/sigmay[i]**2)
        sig0 = sig0+ (1/sigmay[i]**2)
        
    delta = sig0*x_sq -(x_i**2)
    B = (1/delta)*((x_sq*y_i)-(x_i*xy))
    A = (1/delta)*((sig0*xy)-(x_i*y_i))
    sigmaB = (x_sq/delta)**0.5
    sigmA = (sig0/delta)**0.5
    covAB = -x_i/delta

    return [A,B,sigmA,sigmaB,covAB]


#Fit di diretta proporzionalità pesato



#Scarti nel caso di un fit lineare non pesato

def scarti_lin(A,B,x,y,sigma):
    e = []
    for i in range(len(x)):
        e.append((y[i]-A*x[i]+B)/sigma)
    return e
        

#Scarti nel caso di un fit lineare pesato

def w_scarti_lin(A,B,x,y,sigma):
    e = []
    for i in range(len(x)):
        e.append((y[i]-A*x[i]+B)/sigma[i])
    return e


#Scarti nel caso di un fit diretta proporzionalità non pesato

def scarti_dir_prop(M,x,y,sigma):
    e = []
    for i in range(len(x)):
        e.append((y[i]-M*x[i])/sigma)
    return e


#Scarti nel caso di un fit diretta proporzionalità pesato

def w_scarti_dir_prop(M,x,y,sigma):
    e = []
    for i in range(len(x)):
        e.append((y[i]-M*x[i])/sigma[i])
    return e


#Test di ipotesi

def test_hp(x,sigmax,y,sigmay):
    t = (max(sigmax,sigmay)-min(sigmax,sigmay))/((sigmax**2)+(sigmay**2))**0.5
    return t

#Grafico fit lineare

def linear_graph(x,y,A,B):
    plt.scatter(x,y)
    d = np.linspace(min(x),max(x),100)
    plt.plot(d,A*d+b)



#Grafico fit lineare con errori

def w_linear_graph(x,y,sigmax,A,B):
    plt.errorbar(x,y,sigmax,fmt="none")
    d = np.linspace(min(x),max(x),100)
    plt.plot(d,A*d+b)


#Fit Parabolico

def par_fit(x,y,s):
    
    w_i = 1/s**2

    W = sum(w_i)
    X_1 = sum(w_i*x)
    X_2 = sum(w_i*(x**2))
    X_3 = sum(w_i*(x**3))
    X_4 = sum(w_i*(x**4))
    yx_2 = sum(w_i*(x**2)*y)
    yx = sum(w_i*x*y)
    Y = sum(w_i*y)


    G = np.array([[X_4,X_3,X_2],[X_3,X_2,X_1],[X_2,X_1,W]])
    u = np.array([yx_2,yx,Y])
    
    SIGMA = np.linalg.inv(G)

    A,B,C = np.linalg.solve(G,u)
    
    sigmA, covAB,covAC = SIGMA[0]
    _, sigmaB , covBC = SIGMA[1]
    _,_, sigmaC = SIGMA[2]

    return np.array([A,B,C,sigmA,sigmaB,sigmaC,covAB,covBC,covAC])
 
    
#Intersezione tra due rette

def l_cross(A1,B1,A2,B2,sigmA1,sigmaB1,sigmA2,sigmaB2,covAB1,covAB2):
    x = (B2-B1)/(A1-A2)
    y = (A1*B2 - A2*B1)/(A1-A2)
    sigmax = ((((B1-B2)**2)/((A1-A2)**4)*(sigmA1**2 +sigmA2**2)) + (1/((A1-A2)**2) * (sigmaB1**2 + sigmaB2**2)) - ((B1-B2)/((A1-A2)**3))*(covAB1+covAB2))**0.5
    sigmay = (((((B2-B1)**2))/((A1-A2)**4)*((A2*sigmaA1)**2 + (A1*sigmaA2)**2)) + (((A2*sigmaB1)**2) + ()**2) )**0.5
    




