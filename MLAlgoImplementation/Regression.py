import numpy as np 
from matplotlib import pyplot as plt

def reg_function(z, reg_type):
    # returns appropriate function based on type of regression problem
     
    if (reg_type == 'linear'):
        return z
    if (reg_type == 'logistic'):
        return 1/(1+np.exp(-z))

class Regression:

    def __init__(self, reg_type):
        # constructor - takes type of regressions problem (currently 'linear' or 'logistic)
        
        self.reg_type = reg_type
        self.weights = None

    def fit_weights(self, X, y, optimizer = 'sgd', learning_rate = 0.02, max_iter = 15000):
        # uses the optimizer along with parameters selected to fit the model to the data

        assert X.shape[0] == y.shape[0]  # require consistent number of samples
        n_samples = y.shape[0]
        n_features = X.shape[1]
        X = np.insert(X, 0, 1, axis=1)   # insert 1 as the first entry in each sample

        if (optimizer == 'normal'):
            assert self.reg_type == 'linear' # enforce normal equation for linear regression only
            self.weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
            return self.weights

        if (optimizer == 'batch_gd'):
            iter = 0
            self.weights = np.random.randn(X.shape[1],1)
            while (iter < max_iter):
                self.weights = self.weights + learning_rate * np.dot(X.T,(y- reg_function(X @ self.weights, self.reg_type)))
                iter += 1
            return self.weights

        if (optimizer == 'sgd'):
            iter = 0
            self.weights = np.random.randn(X.shape[1],1)
            while (iter < max_iter):
                for i in range(n_samples):
                    hypothesis = reg_function(X @ self.weights, self.reg_type)
                    error = y[i][0] - hypothesis[i][0]
                    self.weights = self.weights + learning_rate * error * X[i].reshape(-1,1)
                iter += 1
            return self.weights 

    def make_prediction(self, X):
        # returns model predictions for given matrix of data X

        X = np.insert(X, 0, 1, axis=1)
        result = reg_function(X @ self.weights, self.reg_type)
        return result

#Linear Test Data

#X = np.random.randn(100,1)
#y = 3 * X + 5 + 3*np.random.randn(100,1)

#test = Regression('linear')
#weights = test.fit_weights(X,y,learning_rate=.001, optimizer='sgd')
#y_pred = test.make_prediction(X)
#print(weights)

#plt.scatter(X,y)
#plt.plot(X,y_pred, color = 'red')
#plt.show()

#Logistic Test Data (from wikipedia)

#X = np.array([0.50,0.75,1.00,1.25,1.50,1.75,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,4.00,4.25,4.50,4.75,5.00,5.50]).reshape(-1,1)
#y = np.array([0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]).reshape(-1,1)
#test = Regression('logistic')
#weights = test.fit_weights(X,y,learning_rate=.002, optimizer='batch_gd')
#y_pred = test.make_prediction(X)
#print(weights)

#plt.scatter(X,y)
#plt.plot(X,y_pred, color = 'red')
#plt.show()