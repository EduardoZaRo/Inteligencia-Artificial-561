'''
a. Diseñe e implemente un modelo de regresión polinomial multivariable de
orden m, usando mínimos cuadrados en lotes OLS Moore-Penrose, con métricas 
de desempeño. Use los datos del archivo synthetic2_dataset.dat para un modelo 
de regresión multivariable de grado 2.
'''
#FALTA LA IMPLEMENTACION POR LOTES
import math
import numpy as np
import designMatrix as dm
import matplotlib.pyplot as plt
import scipy.stats as stats
np.set_printoptions(precision=3) #Para limitar decimales en numpy
def RegressionOLSMoorePenrose(X,Y):
    q,n = X.shape
    #A = np.ones((q,1)) #esto sirve si es regresion lineal
    A = dm.designMatrix(2, X)
    A = np.hstack((A, X))
    ATA = np.matmul(A.T,A)
    b = A.T@Y
    THETA = np.linalg.pinv(ATA)@b
    Yh = A@THETA
    e = Y-Yh
    RMSE = math.sqrt((np.square(e)).mean())
    print("RMSE: ", RMSE)
    return THETA, RMSE, Yh, e
#Cambiar por direccion de nuestra maquina
#Importante checar con que se separan los elementos en el .dat (coma o espacio)

dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_2_3\\{}'.format("synthetic2_dataset.dat"), delimiter = ',')
dataset = np.array(dataset)
X = dataset[:,:-1]  #Toma de la primera hasta la penultima columna
Y = dataset[:,-1:]  #Ultima columna

theta, rmse, Yh, e = RegressionOLSMoorePenrose(X,Y)

#Metricas de desempeno
r, _= stats.pearsonr (Y.T[0], Yh.T[0])
q,n = X.shape   
e = Y-Yh
SSE = e.T@e
MSE = SSE/q
RMSE = np.sqrt(MSE)
INDEX = r/MSE
print("SSE: ", SSE, "\n","MSE: ", MSE, "\n", "RMSE: ", RMSE, "\n","r = : ", r, "\n","INDEX: ", INDEX, "\n",)

#Grafica correlacion Y vs Yh
plt.plot(Y.T[0], Yh.T[0], '.')
m, b = np.polyfit(Y.T[0], Yh.T[0], 1)
plt.plot(Y.T[0], m*Y.T[0] + b)
plt.title("R={}".format(r))
str = "{}*{}+{}".format(round(m,3),"Target",round(b,3))
plt.ylabel(str)
plt.show()

