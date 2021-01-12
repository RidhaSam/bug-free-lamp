import numpy as np 
from matplotlib import pyplot as plt

class NaiveBayes:

    def __init__(self, nb_type = 'bernoulli'):
        self.phi_1 = None
        self.phi_0 = None
        self.phi_y = None
        self.nb_type = nb_type

    def set_params(self, X, y):
        assert X.shape[0] == y.shape[0] 
        n_samples = y.shape[0]
        n_features = X.shape[1]

        num_pos = (np.sum((y == 1), axis = 0))
        self.phi_y = num_pos/n_samples

        self.phi_1 = np.zeros((n_features,1))
        self.phi_0 = np.zeros((n_features,1))

        for j in range(n_features):
            bool_vector_1 = np.logical_and((X[:,j].reshape(-1,1) == 1), (y == 1))
            self.phi_1[j] = (1 + np.sum(bool_vector_1, axis = 0)) / (2 + num_pos)

            bool_vector_0 = np.logical_and((X[:,j].reshape(-1,1) == 1), (y == 0))
            self.phi_0[j] = (1 + np.sum(bool_vector_0, axis = 0)) / (2 + n_samples - num_pos)

        self.phi_0 = self.phi_0.reshape(-1,1)
        self.phi_1 = self.phi_1.reshape(-1,1)

        return self.phi_0, self.phi_1, self.phi_y

    def make_prediction(self, X):
        assert X.shape[0] == y.shape[0] 
        n_samples = X.shape[0]
        n_features = X.shape[1]
        y_pred = np.zeros((n_samples,1))

        for  i in range(n_samples):
            product_pos_probas = 1
            product_neg_probas = 1
            for j in range(n_features):
                if (X[i][j] == 1):
                    product_pos_probas *= self.phi_1[j]
                    product_neg_probas *= self.phi_0[j]
            
            pred_pos = (product_pos_probas * self.phi_y) / (product_pos_probas * self.phi_y + product_neg_probas * (1-self.phi_y))
            pred_neg = (product_neg_probas * (1-self.phi_y))/ (product_pos_probas * self.phi_y + product_neg_probas * (1-self.phi_y))

            if (pred_pos > pred_neg):
                y_pred[i] = 1
            else:
                y_pred[i] = 0

        return y_pred

# Test
#X = np.array([[0,1,0],[1,1,0],[1,1,1],[1,0,0]])
#y = np.array([[1],[0],[1],[0]])
#a = NaiveBayes()
#print(a.set_params(X,y))
#print(a.make_prediction(X))




