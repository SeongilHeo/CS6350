import numpy as np

class ThreeLayerNN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, gamma_0=None, d=None, learning_rate=0.01, custom_weights=None, printdw=False, stochastic=False):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma_0 = gamma_0
        self.d = d
        self.printdw = printdw
        self.stochastic = stochastic
        
        if custom_weights:
            if custom_weights == "zero":
                self.intialize_weights_zero()
            else:
                self.initialize_custom_weights(custom_weights)
        else:
            self.initialize_weights()
    def intialize_weights_zero(self):
        self.W1 = np.zeros((self.input_size , self.hidden_size1-1))
        self.W2 = np.zeros((self.hidden_size1, self.hidden_size2-1))
        self.W3 = np.zeros((self.hidden_size2, self.output_size))

    def initialize_weights(self):
        self.W1 = np.random.randn(self.input_size, self.hidden_size1-1)
        self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2-1)
        self.W3 = np.random.randn(self.hidden_size2, self.output_size)

    def initialize_custom_weights(self, weights):
        self.W1 = np.array(weights['W1'])
        self.W2 = np.array(weights['W2'])
        self.W3 = np.array(weights['W3'])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):

        self.Z1 = np.dot(X, self.W1)
        self.A1 = self.sigmoid(self.Z1)
        self.A1 = np.hstack((np.ones((X.shape[0], 1)), self.A1))

        self.Z2 = np.dot(self.A1, self.W2)
        self.A2 = self.sigmoid(self.Z2)
        self.A2 = np.hstack((np.ones((X.shape[0], 1)), self.A2))
        
        self.Z3 = np.dot(self.A2, self.W3)
        self.A3 = self.Z3

        return self.A3

    def backward(self, X, y, output):

        dZ3 = output - y
        dW3 = np.dot(self.A2.T, dZ3)

        dZ2 = np.dot(dZ3, self.W3.T)[:, 1:] * self.sigmoid_derivative(self.A2[:, 1:])
        dW2 = np.dot(self.A1.T, dZ2)

        dZ1 = np.dot(dZ2, self.W2.T)[:, 1:] * self.sigmoid_derivative(self.A1[:, 1:])
        dW1 = np.dot(X.T, dZ1)

        if self.printdw:
            print(f"Layer3")
            print(dW3)
            print(f"Layer2")
            print(dW2)
            print(f"Layer1")
            print(dW1)
        self.W3 -= self.learning_rate * dW3
        self.W2 -= self.learning_rate * dW2
        self.W1 -= self.learning_rate * dW1

    def train(self, X, y, epochs):
        if self.stochastic:
            losses = []
            for epoch in range(epochs):
                indices = np.arange(X.shape[0])
                np.random.shuffle(indices)
                X = X[indices]
                y = y[indices]

                for t, (xi, yi) in enumerate(zip(X, y)):
                    xi = xi.reshape(1, -1)
                    yi = yi.reshape(1, -1)
                    output = self.forward(xi)
                    self.learning_rate = self.gamma_0 / (1 + (self.gamma_0 / self.d) * t)
                    self.backward(xi, yi, output)
                
                output = self.forward(X)
                loss = np.mean((y - output) ** 2)
                losses.append(loss)
                # print(f"Epoch {epoch}, Loss: {loss:.4f}")
            return losses
            
        else:
            for epoch in range(epochs):
                output = self.forward(X)
                self.backward(X, y, output)
                if epoch % 100 == 0:
                    loss = np.mean((y - output) ** 2)
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)
    
    def evaluate(self, X, y, label="Test"):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"{label} Accuracy: {(1-accuracy) * 100:.2f}%")
        return accuracy