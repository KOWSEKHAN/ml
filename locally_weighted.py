import numpy as np
import matplotlib.pyplot as plt
def local_regression(xo, X, Y, tau):
    X = np.asarray(X)  
    Y = np.asarray(Y)
    weights = np.exp(- (X - xo) ** 2 / (2 * tau ** 2))
    X_w = np.vstack([np.ones_like(X), X]).T  
    W = np.diag(weights) 
    beta = np.linalg.inv(X_w.T @ W @ X_w) @ X_w.T @ W @ Y
    return beta[0] + beta[1] * xo  
def draw(tau):
    X = np.linspace(-3, 3, num=1000)  
    Y = np.log(np.abs(X**2 - 0.4))  
    predictions = np.array([local_regression(xo, X, Y, tau) for xo in X])
    plt.plot(X, Y, color='black')  
    plt.plot(X, predictions, color='red')  
    plt.title(f"Local Regression with Tau={tau}")
    plt.show()
draw(10)
draw(0.1)
draw(0.01)
draw(0.001)
