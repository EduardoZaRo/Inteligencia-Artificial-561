import numpy as np
def designMatrix(t, X):
    q,n = X.shape
    A = np.array([])
    for p in range(1,q+1):  
        M = powerMatrix(t, X[p-1,:])
        if(p == 1):
            A = M
        else:
            A = np.vstack((A, M))
    return A
def powerMatrix(t, V):
    if(V.size == 0 or t == 0):               #if|V| = 0 or t = 0
        return 1                             #M = 1
    else:
        M = np.array([])                     #M = matriz vacia
        Z = V[:-1]                           #Z = V[1, n-1]
        W = V[-1]                            #W = V[n]
        for k in range(t+1):
            #M = [M | powerMatrix(t-k,Z).W^k]
            M = np.hstack((M, np.dot(powerMatrix(t-k, Z),W**k)))
        return M
'''
#EJEMPLO DE USO
#X = np.random.randint(-50,50,(9,1))          #(0,50,(q,n))
q = 9
n = 2
X = np.arange(q*n).reshape(q,n)
t = 2
print("Design matrix: \n",designMatrix(t,X))
'''