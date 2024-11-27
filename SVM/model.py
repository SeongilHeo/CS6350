import numpy as np
from scipy.optimize import minimize

class Primal:
    def __init__(self, schedule="A", 
                        epochs=100, 
                        a=0.1, r=0.1, C=100/873):
        
        self.schedule = schedule
        
        self.epochs = epochs
        self.learning_rate = r
        self.a=a
        self.C = C

        self.weights = None
        self.bias = None

    def train(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):

            index = np.random.randint(n_samples)
            sample_X, sample_y = X[index], y[index]

            subgrad_w, subgrad_b = self._objective(n_samples, self.weights, self.bias, sample_X, sample_y)
            self.learning_rate = self._scheduel(self.learning_rate, epoch)

            self.weights -= self.learning_rate * subgrad_w
            self.bias -= self.learning_rate * subgrad_b



    def _objective(self,N,w,b,x,y):
        hinge_loss = max(0, 1 - y * (np.dot(w, x) + b))

        if hinge_loss==0:
            return w, 0
        else:
            return w - self.C * N * y * x, -self.C * N * y

    def _scheduel(self, r, t): 
        if self.schedule == "A":
            return r / (1 + (r / self.a) * t)
        elif self.schedule == "B":
            return r / (1 +  t)
        
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        error_num = np.sum(y != y_hat)

        return error_num / y.shape[0]

class Dual:
    def __init__(self, kernel=None, C=100/873, gamma=None):
        self.kernel = kernel
        
        self.C = C
        self.gamma = gamma 

        self.K = None
        self.weights = None
        self.alpha = None
        self.bias = None

        self.support_indices = None
        self.support_vectors = None
        self.support_labels = None
        self.support_alpha = None

    def train(self, X, y):
        n_samples = X.shape[0]
        
        self.alpha = np.zeros(n_samples)

        bounds = [(0, self.C) for _ in range(n_samples)]
        
        if self.kernel == "gaussian":
            self.K = self._gaussian_kernel(X, X)* np.outer(y, y)
        else:
            self.K = self._linear_kernel(X)* np.outer(y, y)

        
        constraints = {
            'type': 'eq', 
            'fun': self._equality_constraint,
            'args': (y,)
        }
        
        result = minimize(
            fun=self._objective,
            x0=self.alpha,       
            jac=self._dual_gradient,
            method='SLSQP',         
            bounds=bounds,          
            constraints=constraints 
        )
        
        self.alpha = result.x

        self._calculate_weights_and_bias(X, y)
        
    def _dual_gradient(self, alpha):
        return np.dot(self.K, alpha) - 1

    def _equality_constraint(self, alpha, y):
        return np.dot(alpha, y)

    def _objective(self, alpha):
        return 0.5 * np.dot(alpha, np.dot(self.K, alpha)) - np.sum(alpha)
    
    def _linear_kernel(self, X):
        K = np.dot(X, X.T)
        return K
    
    def _gaussian_kernel(self, X1, X2):
        n_samples_1 = X1.shape[0]
        n_samples_2 = X2.shape[0]
        K = np.zeros((n_samples_1, n_samples_2))

        for i in range(n_samples_1):
            for j in range(n_samples_2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-np.dot(diff, diff) / self.gamma)
        return K
    
    def _calculate_weights_and_bias(self, X, y):
        self.support_indices = np.where(self.alpha > 1e-5)[0]
        self.support_vectors = X[self.support_indices]
        self.support_labels = y[self.support_indices]
        self.support_alpha = self.alpha[self.support_indices]
        
        if self.kernel == "gaussian":
            self.K = self._gaussian_kernel(self.support_vectors, self.support_vectors)
            self.bias = np.mean(self.support_labels - np.dot(self.support_alpha * self.support_labels, self.K))
        else:
            self.weights = np.sum(self.support_alpha[:, None] * self.support_labels[:, None] * self.support_vectors, axis=0)
            self.bias = np.mean(self.support_labels - np.dot(self.support_vectors, self.weights))

    def predict(self, X):
        if self.kernel == "gaussian":
            K = self._gaussian_kernel(X, self.support_vectors)
            return np.sign(np.sum((self.support_alpha * self.support_labels)*K, axis=1) + self.bias)
        else:
            return np.sign(np.dot(X, self.weights) + self.bias)
    
    def evaluate(self, X, y):
        y_hat = self.predict(X)
        error_num = np.sum(y != y_hat)

        return error_num / y.shape[0]
    
