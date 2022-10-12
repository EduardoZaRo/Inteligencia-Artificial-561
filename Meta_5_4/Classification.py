import numpy as np
from designMatrix import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pretty_confusion_matrix import pp_matrix_from_data #IMPORTANTE
from sklearn.metrics import roc_curve
class optimParam:
    epochs = 1000
    goal = 1e-6
    min_grad = 1.0e-6
def plotData(X,Y): #Grafica los datos con colores distintos
    classes = np.unique(Y)
    colors = ['b','g','r','c','m','y'] #Para cambiar de colores
    for i in range(classes.shape[0]):
        plt.plot(X[Y[:,i]==1][:,0], X[Y[:,i]==1][:,1], "o{}".format(colors[i]))
    plt.title("Data")
    plt.show()
def getClasses(Y): 
    if(Y.shape[1] > 1):
        return Y       
    classes = np.unique(Y)
    aux = np.array([])
    counter = 0
    for i in classes:
        if(counter == 0):
            aux = np.array(Y==i)
        else:
            aux = np.hstack((aux, Y == i))
        counter+=1
    Y = aux
    Y = np.array(Y, dtype=int)  
    return Y
def plotROC(Y, ph):
    '''
    -----IMPORTS REQUIRED----
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from matplotlib.pyplot import cm
    '''
    colors = ['b','g','r','c','m','y'] #Para cambiar de colores
    for i in range(Y.shape[1]): #Se iteran las clases
        fpr, tpr, thresholds = roc_curve(Y[:,i].reshape(-1,1), ph[:,i].reshape(-1,1), pos_label=1)
        plt.plot(fpr,tpr, colors[i], label='Class {}'.format(i))
    plt.plot([0,1],[0,1], "k--", label='Random Guess')
    plt.legend(loc="best")
    plt.title("ROC curve")
    plt.show()
def sigmoide(z):    #Funcion sigmoide
    g = np.zeros(z.shape)
    i,j = z.shape
    for a in range(i):
        for b in range(j):
            g[a,b] = 1/(1+np.exp(-1*z[a,b]))
    return g
def logitAvgLoss(A, Y, vecX): #Funcion logistica
    q, col = A.shape
    m = Y.shape[1]
    theta = np.resize(vecX, (col,m))
    hx = A@theta
    P = sigmoide(hx)
    e = Y-P
    #Para evitar log(0) se le suma un valor muy pequeno
    H = Y*np.log(P+1e-12)+(1-Y)*np.log(1-P+1e-12)
    J = -1*np.sum(np.sum(H))/(m*q)
    return J,e
def  logitAvgLossGrad(A,e): #Gradiente funcion logistica
    q = A.shape[0]
    m = e.shape[1]
    grad = -A.T@e / (m*q)
    return grad.flatten()
def logitRegressionNADAM(X,Y, oP, grado):
    q,n = X.shape
    oP.epochs+=1
    m = Y.shape[1]
    A = designMatrix(grado,X)
    theta = np.zeros((A.shape[1], m))
    vecX = theta.flatten().reshape(-1,1) #Se va 
    t_arreglo = np.array([])
    goal_a = np.array([])
    perf_a = np.array([])

    #Desde linea 83 a 113 implementacion de NADAM
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
        perf, e = logitAvgLoss(A,Y,wt)
        gd = logitAvgLossGrad(A,e)
        #vectores anteriores
        mt_gorrito_anterior = mt_gorrito
        #Algoritmo
        mt = beta_1*mt+(1-beta_1)*gd
        vt = beta_2*vt+(1-beta_2)*gd**2
        mt_gorrito = mt/(1-beta_1**(t+1))
        vt_gorrito = vt/(1-beta_2**(t+1))
        wt = wt - (alpha/(np.sqrt(vt_gorrito)+(1e-8)))*(beta_1*mt_gorrito_anterior+((1-beta_1)/(1-beta_1**(t+1)))*gd)
        if(perf <= oP.goal):
            print("Perf goal reached at ", t)
            break
        elif(np.linalg.norm(gd) <  oP.min_grad):
            print("Min grad at ", t)
            break
        elif(t ==  oP.epochs-1):
            print("Max epochs at ", t)
            break
       #print("perf:",perf,"|grad:", np.linalg.norm(gd),"|epoch:",t)
        perf_a = np.append(perf_a, perf)

    #Grafica estatica
    t_arreglo = np.array(range(0,t,1))
    goal_a = np.zeros(t)+oP.goal
    plt.yscale("log")   
    plt.plot(t_arreglo, perf_a, 'b')
    plt.plot(t_arreglo, goal_a, 'r')
    plt.ylabel("Perf")
    plt.xlabel("Epochs")
    plt.show()
    vecX = wt
    print("Perf",perf,"Grad",np.linalg.norm(gd), "Epochs", t)
    thetaHat = np.resize(vecX, (A.shape[1],m))
    return thetaHat, A


opcion = 1

if(opcion == 0):
    dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_4\\{}'.format("syntheticclass_data2.dat"), delimiter = ' ')
    dataset = np.array(dataset)
    X = dataset[:,:-1]
    Y = dataset[:,-1:]
    grado = 6
    Y = getClasses(Y)
if(opcion == 1):
    dataset = np.loadtxt('C:\\Users\\irvin\\Desktop\\Meta_5_4\\{}'.format("syntheticclass_data1.dat"), delimiter = ' ')
    dataset = np.array(dataset)
    X = dataset[:,:-1]
    Y = dataset[:,-1:]
    grado = 1
    Y = getClasses(Y)

n = X.shape[1]
m = Y.shape[1]
oP = optimParam()
thetaHat, A = logitRegressionNADAM(X, Y, oP, grado)
hx = A@thetaHat
ph = sigmoide(hx)
#Confusion matrix without plot
confusion = confusion_matrix(np.argmax(Y, axis=1), np.argmax(ph, axis=1))
print(confusion)

#Plots
plotData(X,Y)
pp_matrix_from_data(np.argmax(Y, axis=1), np.argmax(ph, axis=1))
plotROC(Y,ph)

