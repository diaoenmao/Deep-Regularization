import numpy as np 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

N = 20
X = np.random.uniform(0., 1., size=(20, 2)) 
Y = np.dot(X, [0.45, -0.35]) - 2 + np.random.normal(0, 0.1, 20)

model = LinearRegression().fit(X, Y) 
print(model.score(X, Y)) 
print(model.coef_, model.intercept_)

space = np.linspace(0, 1, 100) 



# plt.scatter(X, Y, c="b", s=2) 
# plt.plot(space, model.coef_ * space + model.intercept_, c="r") 
# plt.show()