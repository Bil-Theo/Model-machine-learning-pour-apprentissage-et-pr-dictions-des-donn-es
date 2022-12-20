import numpy as np
from sklearn.datasets import make_regression 
import matplotlib.pyplot as plt
import myfunctions as fun

#Create of datasets 
x,y = make_regression(n_samples= 100, n_features= 2, noise = 10)
y = y + abs((y/2))

#controle of size for y
print(x.shape)
y = y.reshape(y.shape[0],1)
print(y.shape)

#show the graphe for function of learning
#plt.scatter(x,y)
#plt.show()

#Create X
X = np.hstack((x, np.ones((x.shape[0],1))))

#create vector tetha
tetha = np.random.randn(3,1)

#value final for tetha
tetha_final, cost_History = fun.gradient_Desc(X, y, tetha, learning_rate=0.01, nbr_iterations=400)

#prediction
prediction = fun.modele(X, tetha_final)

R = fun.coef_Determination(y, prediction)
print(R)


fig = plt.figure()
ax3D = fig.add_subplot(111, projection = '3d')
ax3D.scatter(X[:,0], X[:,1], y)
ax3D.scatter(X[:,0], X[:,1], prediction, c="r")
plt.show()

plt.plot(range(400), cost_History)
plt.show()





