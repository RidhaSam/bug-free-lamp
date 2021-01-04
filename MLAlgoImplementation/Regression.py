import numpy as np 
from matplotlib import pyplot as plt

class LinearRegression():

    def __init__(self):
        self.weights = None

    def fit_weights(self, X, y, optimizer = 'normal', learning_rate = 0.02, max_iter = 1000):
        assert X.shape[0] == y.shape[0]
        n_samples = y.shape[0]
        n_features = X.shape[1]
        X = np.insert(X, 0, 1, axis=1)

        if (optimizer == 'normal'):
            self.weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
            return self.weights

        if (optimizer == 'batch_gd'):
            iter = 0
            self.weights = np.random.randn(X.shape[1],1)
            while (iter < max_iter):
                self.weights = self.weights + learning_rate * np.dot(X.T,(y-X @ self.weights))
                iter += 1
            return self.weights

        if (optimizer == 'sgd'):
            iter = 0
            self.weights = np.random.randn(X.shape[1],1)
            while (iter < max_iter):
                for i in range(n_samples):
                    hypothesis = X @ self.weights
                    error = y[i][0] - hypothesis[i][0]
                    self.weights = self.weights + learning_rate * error * X[i].reshape(-1,1)
                iter += 1
            return self.weights

    def make_prediction(self, X):
        X = np.insert(X, 0, 1, axis=1)
        result = X @ self.weights
        return result

## Test with Random Data

#X = np.random.randn(100,1)
#y = 3 * X + 5 + 3*np.random.randn(100,1)

#test = LinearRegression()
#weights = test.fit_weights(X,y,learning_rate=.001, optimizer='batch_gd')
#y_pred = test.make_prediction(X)
#print(weights)

#plt.scatter(X,y)
#plt.plot(X,y_pred, color = 'red')
#plt.show()

    