from ID3 import ID3
from math import log, exp

class AdaBoost:
    """
    A class that implements the AdaBoost algorithm.

    Attributes:
        max_depth (int, optional)
        criterion_func (function)
        root (Node, optional)
        attributes (dict, optional)
        numerical_attributes (list, optional)

        attributes = (dict)
        columns = (dict)
        numerical_attributes (list, optional)

        num_estimators (int, optional)
        alphas (float)
        stumps (list)
        criterion (function)

    Methods:
        train(X, y, labels, attributes, columns, numerical_attributes)
        predict(data)
        evaluate(data, labels, verbose=False)
    """
    def __init__(self, num_estimators = 50, criterion = 'information_gain'):
        """
        Constructor.

        Args:
            num_estimators (int, optional): number of decision tree stump.
            criterion (str, optional): criterion used to evaluate the inpurity of data.
                                       ['information_gain', 'majority_error', 'gini_index'].
                                       (Default:'information_gain').
        """
        self.attributes = None
        self.columns = None
        self.numerical_attributes = None

        self.num_estimators = num_estimators
        self.alphas = []
        self.stumps = []
        self.criterion = criterion 

    def train(self, X, y, labels, attributes, columns, numerical_attributes, test_X=None, test_y=None):
        """
        Train the AdaBoost on the given dataset.

        Args:
            X (list of lists): data samples.
            y (list): labels corresponding to each sample in the `data`.
            labels (list): possible output of sample in the `data`.
            attributes (dict): 'keys': attribute names, 'values': possible values, if numerical attribute maps `None` value 
            columns (dict): 'keys': attribute names, 'values': column index in the dataset.
            numerical_attributes (list, optional): numerical attributes. 

        Returns:
            None:
        """
        self.labels = labels
        self.attributes = attributes
        self.columns = columns
        self.numerical_attributes = numerical_attributes

        # number of sample
        m = len(X)

        # initial set of weights
        D = [ 1 / m for _ in range(m)]
        
        for t in range(self.num_estimators):
            # train decision stump
            stump = ID3(max_depth = 1, criterion = self.criterion)
            stump.train(X, y,
                        D,
                        attributes,
                        columns,
                        numerical_attributes)
            y_pred = stump.predict(X)
            
            # stump.visualization()
            # calculate error
            err = sum([D[i] for i in range(m) if y_pred[i]!= y[i]])
            
            # calcualter alpha
            if err == 0: err = 1e-10
            alpha = 0.5 * log((1 - err) / err)
            self.alphas.append(alpha)
            self.stumps.append(stump)

            # update weight
            for i in range(m):
                D[i] = D[i]*exp(-alpha * 1 if y[i] == y_pred[i] else -1) 
            total_weight = sum(D)
            D = [d/total_weight for d in D]

            print(f"\t[{t}th]{'-'*30}")
            self.evaluate(X,y,verbose=True)
            self.evaluate(test_X,test_y,verbose=True)

    def predict(self, data):
        """
        Predict the labels for the data.
        
        Args: 
            data (list of list): data samples.

        Results:
            list: predicted labels for the `data`.
        """
        # calculate final hypothesis
        m = len(data)
        final_hypothesis = [0] * m
        for alpha, stump in zip(self.alphas, self.stumps):
            stemp_prediction = stump.predict(data)
            for i in range(m):
                final_hypothesis[i] += alpha * 1 if stemp_prediction[i]==self.labels[1] else 0
        return [self.labels[1] if hypothesis > 0 else self.labels[0] for hypothesis in final_hypothesis]
    
    def evaluate(self, data, labels, verbose=False):
        """
        Evaluate the performance of the decision tree on a given dataset.

        Args:
            data (list of lists): data samples.
            labels (list): true labels corresponding to each sample in the `data`.
            verbose (bool, optional): ff `True`, prints detailed information. (Default: `False`).

        Returns:
            int: the number of correctly predicted labels.
        """
        total_cnt = len(labels)
        predictions = self.predict(data)
        result = 0 
        for i in range(total_cnt):
            if predictions[i] == labels[i]:
                result += 1

        ratio = result/total_cnt

        if verbose:
            # print(f"[Info] Out of {total_cnt} data points, {result} match. Accuracy: {ratio*100}%, Error rate: {(1-ratio)*100}%.")
            print(f"[Info] Accuracy: {ratio*100}%, Error rate: {(1-ratio)*100}%.")

        return ratio