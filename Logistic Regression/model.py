import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, max_epochs=100, prior_variance=None):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.prior_variance = prior_variance
        self.d = 100
        self.train_losses = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, X, y):
        pred = self.sigmoid(np.dot(X, self.weights))
        log_loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
        if self.prior_variance:
            regularization = np.sum(self.weights**2) / (2 * self.prior_variance)
            return log_loss + regularization
        return log_loss

    def compute_gradient(self, X, y):
        pred = self.sigmoid(np.dot(X, self.weights))
        gradient = np.dot(X.T, (pred - y)) / len(y)
        if self.prior_variance:
            gradient += self.weights / self.prior_variance
        return gradient

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.train_losses = []
        for epoch in range(self.max_epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            gradient = self.compute_gradient(X, y)
            learning_rate = self.learning_rate / (1 + (self.learning_rate / self.d) * epoch)
            self.weights -= learning_rate * gradient
            
            loss = self.compute_loss(X, y)
            self.train_losses.append(loss)
            # print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)
    
    def evaluate(self, X, y, label):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"{label} error: {(1-accuracy) * 100:.2f}%")