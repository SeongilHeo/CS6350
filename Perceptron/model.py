import numpy as np
try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        # Add 1 and constant feature into x and w for bias
        X = np.c_[X, np.ones((X.shape[0], 1))]
        n_samples, n_features = X.shape
        # Initialize
        self.weights = np.zeros(n_features)
        
        # Convert labels to {-1,1}
        y_ = np.where(y <= 0, -1, 1)

        # Train model
        for t in range(self.epochs):
            # Shuffle the data
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X = X[indices]
            y_ = y_[indices]
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights)
                y_hat = np.sign(linear_output)
                
                # Update weights
                if y_[idx] * y_hat <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i

    def predict(self, X):
        # Add 1 into x for bias
        X = np.c_[X, np.ones((X.shape[0], 1))]
        # Predict y
        linear_output = np.dot(X, self.weights)
        y_hat = np.sign(linear_output)
        # Convert {-1,1} to {0,1}
        return np.where(y_hat <= 0, 0, 1)
    
    def evaluate(self, X, y, verbose= True):
        # Predict y
        y_hat = self.predict(X)
        error = np.mean(y_hat != y)
        if verbose:
            print(f"Average Error: {error}")
            print("-"*70)

        return error
    
    def get_weight(self, verbose=False):
        if verbose:
            print(f"Wegiths: {self.weights}")
            print("-"*70)

        return self.weights
    
class VotedPerceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_history = []
        self.vote_counts = []

    def fit(self, X, y):
        # Add 1 and constant feature into x and w for bias
        X = np.c_[X, np.ones((X.shape[0], 1))]
        n_samples, n_features = X.shape
        # Initialize
        weights = np.zeros(n_features)
        count = 0

        # Convert labels to {-1,1}
        y_ = np.where(y <= 0, -1, 1)

        # Train model
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, weights)
                y_hat = np.sign(linear_output)

                # Update weights
                if y_[idx] * y_hat <= 0:
                    self.weight_history.append(weights.copy())
                    self.vote_counts.append(count)
                    
                    weights += self.learning_rate * y_[idx] * x_i
                    count = 1
                else:
                    count += 1

        self.weight_history.append(weights.copy())
        self.vote_counts.append(count)

    def predict(self, X):
        # Add 1 into x for bias
        X = np.c_[X, np.ones((X.shape[0], 1))]
        # Initialize
        y_hat  = np.zeros(X.shape[0])
        # Predict y
        for weights, vote in zip(self.weight_history, self.vote_counts):
            predictions = np.sign(np.dot(X, weights))
            y_hat += vote * predictions

        # Convert {-1,1} to {0,1}
        return np.where(y_hat <= 0, 0, 1)
    
    def evaluate(self, X, y, verbose= True):
        # Predict y
        y_hat = self.predict(X)
        error = np.mean(y_hat != y)

        if verbose:
            print(f"Average Error: {error}")
            print("-"*70)

        return error
    
    def get_weight(self, verbose=False):
        print(f"Wegiths List:")
        if verbose:
            for i, (weights, count) in enumerate(zip(self.weight_history, self.vote_counts)):
                print(f"{i}:\t {weights}, \t count: {count}")
            print("-"*70)
            
        return self.weight_history
    
class AveragePerceptron:
    def __init__(self, learning_rate=0.01, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.average_weights = None

    def fit(self, X, y):
        # Add 1 and constant feature into x and w for bias
        X = np.c_[X, np.ones((X.shape[0], 1))]
        n_samples, n_features = X.shape

        # Initialize
        self.weights = np.zeros(n_features)
        self.average_weights = np.zeros(n_features)  # 평균 가중치 벡터

        # Convert labels to {-1,1}
        y_ = np.where(y <= 0, -1, 1)

        # Train model
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights)
                y_hat = np.sign(linear_output)

                # Update weights
                if y_[idx] * y_hat <= 0:
                    self.weights += self.learning_rate * y_[idx] * x_i

                # Update average weights
                self.average_weights += self.weights

        # Compute average weights
        self.average_weights /= (self.epochs * n_samples)

    def predict(self, X):
        # Add 1 into x for bias
        X = np.c_[X, np.ones((X.shape[0], 1))]
        # Predict y
        linear_output = np.dot(X, self.average_weights)
        y_hat = np.sign(linear_output)
        return np.where(y_hat <= 0, 0, 1)  # 예측 결과를 0 또는 1로 변환

    def evaluate(self, X, y, verbose= True):
        # Predict y
        y_hat = self.predict(X)
        error = np.mean(y_hat != y)
        
        if verbose:
            print(f"Average Error: {error}")
            print("-"*70)
        return error
    
    def get_weight(self, verbose=False):
        if verbose:
            print(f"Average Wegiths: {self.average_weights}")
            print("-"*70)
        return self.average_weights