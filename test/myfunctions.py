import numpy as np

def modele(X, tetha):
    return X.dot(tetha)

def cost_function(X, y, tetha):
    m = len(y)
    return 1/(2*m )* np.sum((modele(X, tetha) - y)**2)

def gradient(X, y, tetha):
    m = len(y)
    return 1/m * X.T.dot(modele(X, tetha) - y)

def gradient_Desc(X, y, tetha, learning_rate, nbr_iterations):
    cost_History = np.zeros(nbr_iterations)
    
    for i in range(0, nbr_iterations):
        tetha = tetha - learning_rate * gradient(X, y, tetha)
        cost_History[i] = cost_function(X, y, tetha)

    return tetha, cost_History

def coef_Determination(y, prediction):
    u = ((y - prediction)**2).sum()
    v = ((y - y.mean())**2).sum()

    return 1 - u/v

