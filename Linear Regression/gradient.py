import numpy as np
try:
    import warnings
    warnings.filterwarnings('ignore')
except:
    pass

def compute_cost(X, Y, weights):
    Y_hat = X.dot(weights)
    cost = (1/2) * np.sum(np.square(Y-Y_hat))
    return cost

def batch_gradient_descent(X, Y, test_X, test_Y, r=0.01, threshold=1e-6, max_iters=1000):
    weights = np.zeros(X.shape[1]) 
    train_cost_history = []
    test_cost_history = []

    for i in range(max_iters):
        # predict
        Y_hat = X.dot(weights)
        error = Y-Y_hat
        # gradient descent
        gradient =  - X.T.dot(error)
        weights -= r * gradient
        # calcualte cost
        train_cost = compute_cost(X, Y, weights)
        test_cost = compute_cost(test_X, test_Y, weights)
        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
        # check converged
        if i > 0:
            norm = np.max(np.abs(train_cost_history[i] - train_cost_history[i-1]))
            if norm < threshold:
                print(f"[CONVERGED] r: {r}, iteraion: {i}")
                break
    
    return weights, train_cost_history, test_cost_history

def stochastic_gradient_descent(X, Y, test_X, test_Y, r=0.01, threshold=1e-6, max_iters=1000):
    weights = np.zeros(X.shape[1]) 
    train_cost_history = []
    test_cost_history = []
    
    for i in range(max_iters):
        # draw sample 1ec
        idx = np.random.randint(len(Y))
        x_i = X[idx, :].reshape(1, -1)
        y_i = Y[idx].reshape(1)
        # predict
        y_hat = x_i.dot(weights)
        error = y_i - y_hat 
        # gradient descent
        gradient = -x_i.T.dot(error)
        weights -= r * gradient
        # calcualte cost
        train_cost = compute_cost(X, Y, weights)
        test_cost = compute_cost(test_X, test_Y, weights)
        train_cost_history.append(train_cost)
        test_cost_history.append(test_cost)
        # check converged
        if i > 0:
            norm = np.max(np.abs(train_cost_history[i] - train_cost_history[i-1]))
            if norm < threshold:
                print(f"[CONVERGED] r: {r}, iteraion: {i}")
                break
            
    return weights, train_cost_history, test_cost_history

def get_optional_weight(X,Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y