

import numpy as np
import matplotlib.pyplot as plt
def funcion(x):
    # Extended Rosenbrock function 
    # Minima -> f=0 at (1,.....,1) 
    n = len(x) # n even
    fvec = np.zeros((n,1))
    idx1 = np.array(range(0,n,2)) # odd index
    idx2 = np.array(range(1,n,2)) # even index
    fvec[idx1]=10.0*(x[idx2]-(x[idx1])**2.0)
    fvec[idx2]=1.0-x[idx1]
    f = fvec.T @ fvec
    return f[0,0]
def gradiente(x):
    # Extended Rosenbrock gradient function 
    n = len(x) # n even
    Jf = np.zeros((n,n))
    fvec = np.zeros((n,1))
    idx1 = np.array(range(0,n,2)) # odd index
    idx2 = np.array(range(1,n,2)) # even index
    
    fvec[idx1]=10.0*(x[idx2]-(x[idx1])**2.0)
    fvec[idx2]=1.0-x[idx1]
    for i in range(n//2):
        Jf[2*i,2*i]     = -20.0*x[2*i]
        Jf[2*i,2*i+1]   = 10.0
        Jf[2*i+1,2*i]   = -1.0
    gX = np.matmul(2.0*Jf.T,fvec)
    return gX

def nadam(beta_1, beta_2, alpha, tmax, goal, mingrad,eps, n):
   
    
    wt = np.zeros((n,1))+50
    mt = np.zeros((n,1))
    vt = np.zeros((n,1))
    mt_gorrito = np.zeros((n,1))
    vt_gorrito = np.zeros((n,1))
    
    #Para graficar
    t_arreglo = np.array([])
    goal_a = np.array([])
    perf_a = np.array([])
    
    for t in range(tmax):
        gd = gradiente(wt)
        #vectores anteriores
        mt_gorrito_anterior = mt_gorrito
        #Algoritmo
        mt = beta_1*mt+(1-beta_1)*gd
        vt = beta_2*vt+(1-beta_2)*gd**2
        mt_gorrito = mt/(1-beta_1**(t+1))
        vt_gorrito = vt/(1-beta_2**(t+1))
        wt = wt - (alpha/(np.sqrt(vt_gorrito)+eps))*(beta_1*mt_gorrito_anterior+((1-beta_1)/(1-beta_1**(t+1)))*gd)
        #Primera iteracion solo para comprobar el calculo a mano
        if(t==0):
            print("\nPrimera iteracion\nmt:\n:",mt,"\nvt:\n:",vt,"\nmt^:\n:",mt_gorrito,"\nvt^:\n:",vt_gorrito,"\nwt:\n:",wt)
            print("_______________________________")
        perf = funcion(wt)
        if(perf <= goal):
            print("En la iteracion ",t," se alcanzo la precision de ",goal)
            t_arreglo = np.array(range(0,t,1))
            goal_a = np.zeros(t)+goal
            break
        elif(np.linalg.norm(gd) < mingrad):
            print("En la iteracion ",t," se alcanzo el gradiente minimo de ",mingrad)
            t_arreglo = np.array(range(0,t,1))
            goal_a = np.zeros(t)+goal
            break
        elif(t == tmax-1):
            print("Se alcanzo la maxima cantidad de iteraciones, la meta no se consiguio :(")
            t_arreglo = np.array(range(0,t,1))
            goal_a = np.zeros(t)+goal
            break
        perf_a = np.append(perf_a, perf)
    
    plt.yscale("log")   
    plt.plot(t_arreglo, goal_a)
    plt.plot(t_arreglo, perf_a)
    plt.show()
    return wt
beta_1 = 0.9
beta_2 = 0.999
alpha = 0.01
tmax = 100000
goal = 10**-8
mingrad = 10**-10
eps = 10**-8
n = 100

wt = nadam(beta_1, beta_2, alpha, tmax, goal, mingrad,eps, n)
print(wt[0])
print("f(x) final:\n",funcion(wt))


