import math
import numpy as np
import designMatrix as dm
import matplotlib.pyplot as plt
import scipy.stats as stats
np.set_printoptions(precision=3) #Para limitar decimales en numpy
#Simulacion de variables estaticas
def P(x):
    P.p = x
def Theta(x):
    Theta.t = x
def RegressionRLS(X,Y):
    q,n = X.shape    
    A = dm.designMatrix(1,X)
    P.p = np.zeros((X.shape[0], Y.shape[1]))
    Theta.t = np.zeros((X.shape[1], Y.shape[1]))
    for p in range(q):
        xIn = A[p,:]
        xOut = Y[p,:]
        THETA = rls(p,xIn,xOut)
    return THETA, A
def rls(k,a,y):
    npk = a.size
    m = y.size
    if(k == 0):
        Theta(np.zeros((npk,m)))
        P(1e10*np.eye(npk, npk))
    tmp1 = (P.p@a.reshape(-1,1))*(a@P.p)
    tmp2 = 1+(a@P.p)@a.T
    P(P.p-tmp1/tmp2)
    diff = y-a@Theta.t
    tmp3 = P.p@a.reshape(-1,1)
    Theta(Theta.t + tmp3*diff)
    return Theta.t
#Cambiar por direccion de nuestra maquina
#Importante checar con que se separan los elementos en el .dat (coma o espacio)
dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_2_3\\{}'.format("synthetic2_dataset.dat"), delimiter = ',')

dataset = np.array(dataset)
X = dataset[:,:-1]  #Toma de la primera hasta la penultima columna
Y = dataset[:,-1:]  #Ultima columna
thetaHat, A = RegressionRLS(X,Y)
Yh = A@thetaHat

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