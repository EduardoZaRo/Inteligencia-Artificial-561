'''
2. Diseñar modelos de regresión a partir de los datos 
(reaction_dataset.dat) que establecen la relación
entre la velocidad de reacción (targets, última columna 
de la tabla) y concentraciones de tres reactivos
como medidas predictoras (inputs, primeras tres columnas 
de la tabla): x1(hidrógeno), x2(n-pentano),
x3(isopentano). La velocidad de reacción (y^) para la 
cinética de reacción es el modelo de Hougen-
Watson:
        
    ^          01x2 - x3/05
    y =  _______________________
         1 + 02x1 + 03x2 + 04x3
         
donde Y^ es la velocidad de reacción estimada, 
x1,x2,x3 son las concentraciones de hidrógeno, n-
pentano e isopentano, respectivamente, y 
01,02, ... , 05 son parámetros del modelo.
    
EQUIPO 5:
    HUERTAS VILLEGAS CESAR
    URIAS VEGA JUAN DANIEL
    ZAVALA ROMAN IRVIN EDUARDO
'''
import numpy as np
import designMatrix as dm
import matplotlib.pyplot as plt
import scipy.stats as stats
np.set_printoptions(precision=3)
def funcionMSE(X,Y,theta):
    yh = funcion(X,theta)
    e = Y - yh
    SSE = e.T@e
    MSE = SSE/len(Y)
    return MSE, e
def funcion(x, theta): #Modelo de Hougen-Watson
    x1 = x[:,0].reshape(-1,1)
    x2 = x[:,1].reshape(-1,1)
    x3 = x[:,2].reshape(-1,1)
    yh = (theta[0]*x2 - (x3/theta[4]))/(1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3)
    return yh
def gradiente(x, y,  theta):#Gradiente del modelo de Hougen-Watson
    x1 = x[:,0].reshape(-1,1)
    x2 = x[:,1].reshape(-1,1)
    x3 = x[:,2].reshape(-1,1)
    dyh_d01 = x2/(1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3)
    dyh_d02 = -1*((theta[0]*theta[4]*x2-x3)*x1)/(theta[4]*(1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3))
    dyh_d03 = -1*((theta[0]*theta[4]*x2-x3)*x2)/(theta[4]*(1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3))
    dyh_d04 = -1*((theta[0]*theta[4]*x2-x3)*x3)/(theta[4]*(1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3))
    dyh_d05 = x3/((theta[4]**2)*(1 + theta[1]*x1 + theta[2]*x2 + theta[3]*x3))
    J = -1*np.hstack((dyh_d01,dyh_d02,dyh_d03,dyh_d04,dyh_d05))
    perf, e = funcionMSE(x, y, theta)
    gd = (2.0*J.T)@e
    return gd
def nadam(X, Y, theta):
    wt = theta #Se van a optimizar los parametros theta
    n = len(theta)
    mt = np.zeros((n,1))
    vt = np.zeros((n,1))
    mt_gorrito = np.zeros((n,1))
    vt_gorrito = np.zeros((n,1))

    #Para graficar
    t_arreglo = np.array([])
    goal_a = np.array([])
    perf_a = np.array([])
    beta_1 = 0.975
    beta_2 = 0.999
    alpha = 0.01
    tmax = 20000
    goal = 1e-6
    mingrad = 1e-6
    
    for t in range(tmax):
        perf, e = funcionMSE(X,Y,wt)
        gd =  gradiente(X,Y,wt)
        #vectores anteriores
        mt_gorrito_anterior = mt_gorrito
        #Algoritmo
        mt = beta_1*mt+(1-beta_1)*gd
        vt = beta_2*vt+(1-beta_2)*gd**2
        mt_gorrito = mt/(1-beta_1**(t+1))
        vt_gorrito = vt/(1-beta_2**(t+1))
        wt = wt - (alpha/(np.sqrt(vt_gorrito)+1e-8))*(beta_1*mt_gorrito_anterior+((1-beta_1)/(1-beta_1**(t+1)))*gd)
        
        if(perf <= goal):
            print("En la iteracion ",t," se alcanzo la precision de ",goal)

            break
        elif(np.linalg.norm(gd) < mingrad):
            print("En la iteracion ",t," se alcanzo el gradiente minimo de ",mingrad)

            break
        elif(t == tmax-1):
            print("Se alcanzo la maxima cantidad de iteraciones, la meta no se consiguio :(")
            break
        perf_a = np.append(perf_a, perf)
    print("gd:",np.linalg.norm(gd), "\nperf:",perf)
    t_arreglo = np.arange(t)
    goal_a = np.zeros(t)+goal
    plt.yscale("log")   
    plt.plot(t_arreglo, perf_a, 'b')
    plt.plot(t_arreglo, goal_a, 'r')
    plt.ylabel("Perf")
    plt.xlabel("Epochs")
    plt.show()
    return wt


dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_5\\{}'.format("reaction_dataset.dat"), delimiter = ',')
dataset = np.array(dataset)
theta = np.array([[1], #Theta de inicio ya que no se tiene idea del valor
                  [1],
                  [1],
                  [1],
                  [1]])
X = dataset[:,:-1]
Y = dataset[:,-1:]
theta = nadam(X,Y,theta)
Yh = funcion(X, theta)
print(theta,"\n\tY\t\t\tYh\n",np.hstack((Y,Yh)))
plt.plot(X,Y, 'or')
plt.plot(X,Yh, '*b')
plt.title("Match Y(red) vs Yh(blue)")
plt.show()
print("Y-Yh:\n",Y-Yh)
r, _= stats.pearsonr(Y.T[0], Yh.T[0])
plt.plot(Y.T[0], Yh.T[0], '.')
m, b = np.polyfit(Y.T[0], Yh.T[0], 1)
plt.plot(Y.T[0], m*Y.T[0] + b)
plt.title("R={}".format(r))
str = "{}*{}+{}".format(round(m,3),"Target",round(b,3))
plt.ylabel(str)
plt.show()

#Distintas pruebas
prueba = np.array([[470,300,10]])
yh_prueba = funcion(prueba, theta)
print(yh_prueba) #Deberia aproximar 8.55

prueba = np.array([[285,80,10]])
yh_prueba = funcion(prueba, theta)
print(yh_prueba) #Deberia aproximar 3.79

prueba = np.array([[285,190,120]])
yh_prueba = funcion(prueba, theta)
print(yh_prueba) #Deberia aproximar 3.13

