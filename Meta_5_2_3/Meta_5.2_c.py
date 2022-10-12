import numpy as np
import designMatrix as dm
import matplotlib.pyplot as plt
import scipy.stats as stats
np.set_printoptions(precision=3) #Para limitar decimales en numpy
class optimParam:
    epochs = 0
    goal = 0
    min_grad = 0
    show = 0
def RegressionGradMSE(A,e): #
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

    #Desde linea 36 a 62 implementacion de NADAM
    wt = vecX
    mt =  np.zeros((wt.shape[0], 1))
    vt = np.zeros((wt.shape[0], 1))
    mt_gorrito = np.zeros((wt.shape[0], 1))
    vt_gorrito = np.zeros((wt.shape[0], 1))
    beta_1 = 0.975
    beta_2 = 0.999
    alpha = 0.1
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
    #Grafica dinamica
    for i in range(0,t,t//6):
        t_arreglo = np.array(range(0,i,1))
        goal_a = np.zeros(i)+oP.goal
        plt.yscale("log")   
        plt.plot(t_arreglo, perf_a[:i], 'b')
        plt.plot(t_arreglo, goal_a[:i], 'r')
        plt.ylabel("Perf")
        plt.xlabel("Epochs")
        plt.pause(0.1)
    vecX = wt
    t_arreglo = np.arange(t)
    goal_a = np.zeros(t)+oP.goal
    plt.yscale("log")   
    plt.plot(t_arreglo, perf_a, 'b')
    plt.plot(t_arreglo, goal_a, 'r')
    plt.ylabel("Perf")
    plt.xlabel("Epochs")
    plt.show()
    
    print("Perf",perf,"Grad",gd[0][0], "Epochs", t)
    Theta = np.resize(vecX, (A.shape[1],m))
    return Theta, A



dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_2_3\\{}'.format("synthetic2_dataset.dat"), delimiter = ',')
dataset = np.array(dataset)
X = dataset[:,:-1]  #Toma de la primera hasta la penultima columna
Y = dataset[:,-1:]  #Ultima columna
oP = optimParam()
oP.epochs = 100000
oP.goal = 1e-8
oP.min_grad = 1e-10
oP.show = 20

n = X.shape[1]
m = Y.shape[1]
#theta = np.random.rand(n+1, m)

grado = 2
thetaHat,A = RegressionOptimgd(X,Y,grado,oP)
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
