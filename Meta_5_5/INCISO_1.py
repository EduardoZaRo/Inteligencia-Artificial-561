'''
1. Diseñar modelos de regresión a partir de los datos 
(bodyfat_dataset.dat) que establecen la relación entre 
la masa corporal (targets, última columna de la tabla) 
de 252 pacientes a partir de 13 medidas predictoras 
(inputs, primeras 13 columnas de la tabla) : Edad (años), 
Peso (libras), Altura (pulgadas),Circunferencia del cuello (cm),
Circunferencia del pecho (cm), Circunferencia del abdomen 2 (cm),
Circunferencia de la cadera (cm), Circunferencia del muslo (cm), 
Circunferencia de la rodilla (cm),Circunferencia del tobillo (cm),
Circunferencia de bíceps (extendida) (cm), Circunferencia del
antebrazo (cm) y Circunferencia de la muñeca (cm).
El objetivo del análisis de regresión es una forma de técnica 
de modelado predictivo que investiga la relación entre una 
variable dependiente (objetivo) y una variable independiente 
(predictor). Esta técnica se utiliza para pronosticar y 
encontrar la relación del efecto causal entre las variables.
EQUIPO 5:
    HUERTAS VILLEGAS CESAR
    URIAS VEGA JUAN DANIEL
    ZAVALA ROMAN IRVIN EDUARDO
'''
import math
import numpy as np
import designMatrix as dm
import matplotlib.pyplot as plt
import scipy.stats as stats
np.set_printoptions(precision=3) #Para limitar decimales en numpy
#DE LINEAS 30 A 100 ES REGRESION CON OPTIMIZACION NUMERICA
class optimParam:
    epochs = 0
    goal = 0
    min_grad = 0
    show = 0
def RegressionGradMSE(A,e):
    q = A.shape[0]
    m = e.shape[1]
    gradMse = -2*A.T@e / (m*q)
    return gradMse
def RegressionMSE(A,Y,vecX):
    col = A.shape[1]
    m = Y.shape[1]
    Theta = np.resize(vecX, (col,m))
    Yh = A@Theta
    e = Y-Yh
    MSE = (np.square(e)).mean()
    return MSE, e
def RegressionOptimgd(X,Y,grado,oP):
    q,n = X.shape
    oP.epochs+=1
    m = Y.shape[1]
    A = dm.designMatrix(grado,X)
    vecX = np.random.rand(A.shape[1], m)
    #Para graficar
    t_arreglo = np.array([])
    goal_a = np.array([])
    perf_a = np.array([])


    wt = vecX
    
    mt =  np.zeros((wt.shape[0], 1))
    vt = np.zeros((wt.shape[0], 1))
    mt_gorrito = np.zeros((wt.shape[0], 1))
    vt_gorrito = np.zeros((wt.shape[0], 1))
    beta_1 = 0.975
    beta_2 = 0.999
    alpha = 0.01
    oP.epochs+=1
    for t in range(oP.epochs):
        perf, e = RegressionMSE(A,Y,wt)
        gd = RegressionGradMSE(A,e)
        #vectores anteriores
        mt_gorrito_anterior = mt_gorrito
        #Algoritmo
        mt = beta_1*mt+(1-beta_1)*gd
        vt = beta_2*vt+(1-beta_2)*gd**2
        mt_gorrito = mt/(1-beta_1**(t+1))
        vt_gorrito = vt/(1-beta_2**(t+1))
        wt = wt - (alpha/(np.sqrt(vt_gorrito)+1e-8))*(beta_1*mt_gorrito_anterior+((1-beta_1)/(1-beta_1**(t+1)))*gd)
        if(perf <= oP.goal):
            break
        elif(np.linalg.norm(gd) <  oP.min_grad):
            break
        elif(t ==  oP.epochs-1):
            break
        perf_a = np.append(perf_a, perf)
    vecX = wt
    '''t_arreglo = np.arange(t)
    goal_a = np.zeros(t)+oP.goal
    plt.yscale("log")   
    plt.plot(t_arreglo, perf_a, 'b')
    plt.plot(t_arreglo, goal_a, 'r')
    plt.ylabel("Perf")
    plt.xlabel("Epochs")
    plt.show()'''
    
    print("Perf",perf,"Grad",gd[0][0], "Epochs", t)
    Theta = np.resize(vecX, (A.shape[1],m))
    return Theta, A

#DE LINEAS 103 A 115 ES OLS
def RegressionOLSMoorePenrose(X,Y):
    q,n = X.shape
    #A = np.ones((q,1)) #esto sirve si es regresion lineal
    A = dm.designMatrix(1, X)
    A = np.hstack((A, X))
    ATA = np.matmul(A.T,A)
    b = A.T@Y
    THETA = np.linalg.pinv(ATA)@b
    Yh = A@THETA
    e = Y-Yh
    RMSE = math.sqrt((np.square(e)).mean())
    print("RMSE: ", RMSE)
    return THETA, RMSE, Yh, e

#DE LINEAS 118 A 144 ES RLS
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

def desempeno(X, Y, Yh, string):
    r, _= stats.pearsonr (Y.T[0], Yh.T[0])
    q,n = X.shape   
    e = Y-Yh
    SSE = e.T@e
    MSE = SSE/q
    RMSE = np.sqrt(MSE)
    INDEX = r/MSE
    print("SSE: ", SSE, "\n","MSE: ", MSE, "\n", "RMSE: ", RMSE, "\n","r = : ", r, "\n","INDEX: ", INDEX, "\n",)

    plt.plot(Y.T[0], Yh.T[0], '.')
    m, b = np.polyfit(Y.T[0], Yh.T[0], 1)
    plt.plot(Y.T[0], m*Y.T[0] + b)
    plt.title("R={} Metodo: {}".format(r, string))
    str = "{}*{}+{}".format(round(m,3),"Target",round(b,3))
    plt.ylabel(str)
    plt.show()
dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_5\\{}'.format("bodyfat_dataset.dat"), delimiter = '\t')
dataset = np.array(dataset)
X = dataset[:,:-1]
Y = dataset[:,-1:]
oP = optimParam()
oP.epochs = 10000
oP.goal = 1e-8
oP.min_grad = 1e-10
oP.show = 20

n = X.shape[1]
m = Y.shape[1]

grado = 1
thetaHat,A = RegressionOptimgd(X,Y,grado,oP)
Yh1 = A@thetaHat 
theta, rmse, Yh2, e = RegressionOLSMoorePenrose(X,Y)
thetaHat, A = RegressionRLS(X,Y)
Yh3 = A@thetaHat
desempeno(X, Y,Yh1, "Optimizacion numerica")
desempeno(X,Y,Yh2, "OLS")
desempeno(X,Y,Yh3, "RLS")

'''
SE PUEDE CONCLUIR QUE EL METODO OLS DA LA MEJOR CORRELACION
AUNQUE CON GRADO 2, LOS DEMAS CON GRADOS MAYORES A 1 BAJAN
EL RENDIMIENTO
'''




