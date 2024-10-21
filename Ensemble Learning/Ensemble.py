from ID3 import ID3
from math import log, exp
from utils import bootstrap_sample, covert_labels


class AdaBoost:
    """
    A class that implements the AdaBoost algorithm.

    Attributes:
        attributes = (dict)
        columns = (dict)
        numerical_attributes (list, optional)

        num_estimators (int, optional)
        alphas (float)
        stumps (list)
        criterion (function)

    Methods:
        train(X, Y, labels, attributes, columns, numerical_attributes)
        predict(X)
        evaluate(X, Y, verbose=False)
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
        self.labels = None

        self.num_estimators = num_estimators
        self.alphas = []
        self.stumps = []
        self.criterion = criterion 

    def train(self, X, Y, labels, attributes, columns, numerical_attributes, test_X=None, test_Y=None):
        """
        Train the AdaBoost on the given dataset.

        Args:
            X (list of lists): data samples.
            Y (list): labels corresponding to each sample in the `X`.
            labels (list): possible output of sample in the `X`.
            attributes (dict): 'keys': attribute names, 'values': possible values, if numerical attribute maps `None` value 
            columns (dict): 'keys': attribute names, 'values': column index in the dataset.
            numerical_attributes (list, optional): numerical attributes. 

            test_X (list): test data sampel
            test_Y (list): labels corresponding to each sample in the `test_X`

        Returns:
            None:
        """
        self.labels = labels
        self.attributes = attributes
        self.columns = columns
        self.numerical_attributes = numerical_attributes
        
        # label check
        if set(labels) != set([1,-1]):
            Y = covert_labels(Y)
            if test_Y:
                test_Y = covert_labels(test_Y)
            self.labels = [1,-1]

        # for record
        if test_X:
            res = [0] * self.num_estimators

        # number of sample
        m = len(X)
        # initial set of weights
        D = [ 1 / m for _ in range(m)]

        for t in range(self.num_estimators):
            # train decision stump
            stump = ID3(max_depth = 1, criterion = self.criterion)
            stump.train(X, Y,
                        attributes,
                        columns,
                        numerical_attributes,
                        weights=D)
            H = stump.predict(X)

            # calcualte error
            err = sum([D[i] for i in range(m) if Y[i]!=H[i]])

            # calcualte alpha
            alpha = (1/2) * log((1 - err) / err)

            self.alphas.append(alpha)
            self.stumps.append(stump)

            # update sample weights D
            for i in range(m):
                D[i] = D[i] * exp(-1 * alpha * Y[i] * H[i]) 
                # D[i] = D[i] * exp(-1* alpha * (1 if Y[i] == H[i] else -1)) 
            total_weight=sum(D)
            for i in range(m):
                D[i] = D[i]/total_weight
            print(f"{'-'*50}[{t+1}th]")
            # evaluate train and test dataset
            print(f"{'-'*65}[{t+1}th]")
            # evaluate train and test dataset
            if test_X:
                print("[TRAIN]",end=" ")
                train_acc = self.evaluate(X,Y,verbose=True)        
                print("[TEST] ",end=" ")
                test_acc = self.evaluate(test_X,test_Y,verbose=True)
                res[t-1]=[t,train_acc,test_acc]
        if test_X:
            try:
                import pandas as pd
                import datetime
                current_time = datetime.datetime.now()
                pd.DataFrame(res).to_csv(f"./csv/ada_{current_time.strftime('%Y%m%d_%H%M%S')}.csv")
                print(f"[Info] The csv are saved at ./csv/ada_{current_time.strftime('%Y%m%d_%H%M%S')}.csv")
            except:
                pass
            

    def predict(self, X):
        """
        Predict the labels for the data.
        
        Args: 
            X (list of list): data samples.

        Results:
            list: predicted labels for the `X`.
        """
        # calculate final hypothesis
        m = len(X)
        H_final = [0] * m
        for alpha, stump in zip(self.alphas, self.stumps):
            H = stump.predict(X)
            for i in range(m):
                H_final[i] += alpha * H[i]
        
        return [-1 if h < 0 else 1 for h in H_final]
    
    def evaluate(self, X, Y, verbose=False):
        """
        Evaluate the performance of the decision tree on a given dataset.

        Args:
            X (list of lists): data samples. 
            Y (list): true labels corresponding to each sample in the `X`.
            verbose (bool, optional): ff `True`, prints detailed information. (Default: `False`).

        Returns:
            int: the number of correctly predicted labels.
        """
        # label check
        if set(Y) != set([1,-1]):
            Y = covert_labels(Y)

        total_cnt = len(Y)
        Y_hat = self.predict(X)
        result = sum(1 for i in range(total_cnt) if Y_hat[i] == Y[i])
        ratio = result/total_cnt

        if verbose:
            print(f"[EVAL] Accuracy: {ratio*100:.2f}%, Error rate: {(1-ratio)*100:.2f}%.")

        return ratio


class BaggedTrees:
    """
    A class that implements the BaggedTrees algorithm.

    Attributes:
        attributes (dict)
        columns (dict)
        numerical_attributes (list)
        labels (list)

        num_trees (int)
        trees (object)
        criterion (function)

        sample_ratio (float)
        num_sample (int)
        repalce (bool)
    Methods:
        train(X, Y, labels, attributes, columns, numerical_attributes)
        predict(X)
        evaluate(X, Y, verbose=False)
    """
    def __init__(self, num_trees = 50, criterion = 'information_gain', sample_ratio=1, repalce = True):
        """
        Constructor.

        Args:
            num_trees (int, optional): number of decision tree.
            criterion (str, optional): criterion used to evaluate the inpurity of data.
                                       ['information_gain', 'majority_error', 'gini_index'].
                                       (Default:'information_gain').
            sample_ratio (float, optional): bootstrap ratio (Default: 1 (full)).
            replace (bool, optional): boostraping with replacement or not (Default: True).

        """
        self.attributes = None
        self.columns = None
        self.numerical_attributes = None
        self.labels = None

        self.num_trees = num_trees
        self.trees = []
        self.criterion = criterion 
        self.sample_ratio = sample_ratio
        self.num_sample = None
        self.repalce = repalce

    def train(self, X, Y, labels, attributes, columns, numerical_attributes, test_X=None, test_Y=None):
        """
        Train the BaggedTree on the given dataset.

        Args:
            X (list of lists): data samples.
            Y (list): labels corresponding to each sample in the `data`. 
            labels (list): possible output of sample in the `data`.
            attributes (dict): 'keys': attribute names, 'values': possible values, if numerical attribute maps `None` value 
            columns (dict): 'keys': attribute names, 'values': column index in the dataset.
            numerical_attributes (list, optional): numerical attributes. 

            test_X (list, optional): test data sampel
            test_Y (list, optional): labels corresponding to each sample in the `test_X`
        Returns:
            None:
        """
        self.labels = labels
        self.attributes = attributes
        self.columns = columns
        self.numerical_attributes = numerical_attributes
        self.num_sample = int(len(Y) * self.sample_ratio)
        
        # for record
        if test_X:
            res = [0] * self.num_trees

        for t in range(self.num_trees):
            # draw samples
            bootstrap_X, bootstrap_Y = bootstrap_sample(X,Y, num_sample = self.num_sample, replace=self.repalce)

            # single tree
            tree = ID3(criterion = self.criterion)
            tree.train(bootstrap_X, bootstrap_Y, 
                        attributes,
                        columns,
                        numerical_attributes=numerical_attributes)
            self.trees.append(tree)

            print(f"{'-'*65}[{t+1}th]")
            # evaluate train and test dataset
            if test_X:
                print("[TRAIN]",end=" ")
                train_acc = self.evaluate(X,Y,verbose=True)        
                print("[TEST] ",end=" ")
                test_acc = self.evaluate(test_X,test_Y,verbose=True)
                res[t-1]=[t,train_acc,test_acc]
        if test_X:
            try:
                import pandas as pd
                import datetime
                current_time = datetime.datetime.now()
                pd.DataFrame(res).to_csv(f"./csv/bag_{current_time.strftime('%Y%m%d_%H%M%S')}.csv")
                print(f"[Info] The csv are saved at ./csv/bag_{current_time.strftime('%Y%m%d_%H%M%S')}.csv")
            except:
                pass
            
    def predict(self, X):
        """
        Predict the labels for the data.
        
        Args: 
            X (list of list): data samples.

        Results:
            list: predicted labels for the `X`.
        """
        # vote tree's prediction
        m = len(X)
        Y_hat = [0] * m
        Y_hat_vote = [[0] * len(self.labels) for _ in range(m)]
        for tree in self.trees:
            temp_Y_hat = tree.predict(X)
            for i in range(m):
                if  temp_Y_hat[i] == self.labels[0]:
                    Y_hat_vote[i][0]+=1
                else:
                    Y_hat_vote[i][1]+=1
        # agrregate
        for i in range(m):
            Y_hat[i] = self.labels[0] if Y_hat_vote[i][0] > Y_hat_vote[i][1] else self.labels[1]

        return Y_hat

    def evaluate(self, X, Y, verbose=False):
        """
        Evaluate the performance of the decision tree on a given dataset.

        Args:
            X (list of lists): data samples.
            Y (list): true labels corresponding to each sample in the `data`.
            verbose (bool, optional): ff `True`, prints detailed information. (Default: `False`).

        Returns:
            int: the number of correctly predicted labels.
        """
        total_cnt = len(Y)
        Y_hat = self.predict(X)
        correct = 0 
        for i in range(total_cnt):
            if Y_hat[i] == Y[i]:
                correct += 1

        ratio = correct/total_cnt
 
        if verbose:
            print(f"[EVAL] Accuracy: {ratio*100:.2f}%, Error rate: {(1-ratio)*100:.2f}%.")

        return ratio

class RandomForest:
    """
    A class that implements the RandomForest algorithm.

    Attributes:
        attributes = (dict)
        columns = (dict)
        numerical_attributes (list)

        num_trees (int)
        trees (object)
        criterion (function)

        sample_ratio (float)
        num_sample (int)
        repalce (bool)
        num_attributes (int)
    Methods:
        train(X, Y, labels, attributes, columns, numerical_attributes)
        predict(X)
        evaluate(X, Y, verbose=False)
    """
    def __init__(self, num_trees = 50, criterion = 'information_gain', sample_ratio=1, repalce = True, num_attributes=2):
        """
        Constructor.

        Args:
            num_trees (int, optional): number of decision tree.
            criterion (str, optional): criterion used to evaluate the inpurity of data.
                                       ['information_gain', 'majority_error', 'gini_index'].
                                       (Default:'information_gain').
            sample_ratio (float, optional): bootstrap ratio (Default: 1 (full)).
            replace (bool, optional): boostraping with replacement or not (Default: True).
            num_attributes (int, optional): number of attributes to bootstrap (Default: 2).

        """
        self.attributes = None
        self.columns = None
        self.numerical_attributes = None

        self.num_trees = num_trees
        self.trees = []
        self.criterion = criterion 
        self.sample_ratio = sample_ratio
        self.num_sample = None
        self.repalce = repalce
        self.num_attributes = num_attributes

    def train(self, X, Y, labels, attributes, columns, numerical_attributes, test_X=None, test_Y=None):
        """
        Train the BaggedTree on the given dataset.

        Args:
            X (list of lists): data samples.
            Y (list): labels corresponding to each sample in the `data`. 
            labels (list): possible output of sample in the `data`.
            attributes (dict): 'keys': attribute names, 'values': possible values, if numerical attribute maps `None` value 
            columns (dict): 'keys': attribute names, 'values': column index in the dataset.
            numerical_attributes (list, optional): numerical attributes. 

            test_X (list, optional): test data sampel
            test_Y (list, optional): labels corresponding to each sample in the `test_X`
        Returns:
            None:
        """
        self.labels = labels
        self.attributes = attributes
        self.columns = columns
        self.numerical_attributes = numerical_attributes
        self.num_sample = int(len(Y) * self.sample_ratio)

        # for record
        if test_X:
            res = [0] * self.num_trees
            
        for t in range(self.num_trees):
            bootstrap_X, bootstrap_Y  = bootstrap_sample(X, Y, num_sample = self.num_sample)
            tree = ID3(criterion = self.criterion, num_attributes=self.num_attributes)
            tree.train(bootstrap_X, bootstrap_Y, 
                        attributes,
                        columns,
                        numerical_attributes=numerical_attributes)
            self.trees.append(tree)

            print(f"{'-'*65}[{t+1}th]")
            # evaluate train and test dataset
            if test_X:
                print("[TRAIN]",end=" ")
                train_acc = self.evaluate(X,Y,verbose=True)        
                print("[TEST] ",end=" ")
                test_acc = self.evaluate(test_X,test_Y,verbose=True)
                res[t-1]=[t,train_acc,test_acc]
        if test_X:
            try:
                import pandas as pd
                import datetime
                current_time = datetime.datetime.now()
                pd.DataFrame(res).to_csv(f"./csv/random_{current_time.strftime('%Y%m%d_%H%M%S')}.csv")
                print(f"[Info] The csv are saved at ./csv/random_{current_time.strftime('%Y%m%d_%H%M%S')}.csv")
            except:
                pass
            
    def predict(self, X):
        """
        Predict the labels for the data.
        
        Args: 
            X (list of list): data samples.

        Results:
            list: predicted labels for the `X`.
        """
        m = len(X)
        Y_hat = [0] * m
        Y_hat_vote = [[0] * len(self.labels) for _ in range(m)]
        for tree in self.trees:
            temp_Y_hat = tree.predict(X)
            for i in range(m):
                if  temp_Y_hat[i] == self.labels[0]:
                    Y_hat_vote[i][0]+=1
                else:
                    Y_hat_vote[i][1]+=1
        # agrregate
        for i in range(m):
            Y_hat[i] = self.labels[0] if Y_hat_vote[i][0] > Y_hat_vote[i][1] else self.labels[1]

        return Y_hat

    def evaluate(self, X, Y, verbose=False):
        """
        Evaluate the performance of the decision tree on a given dataset.

        Args:
            X (list of lists): data samples.
            Y (list): true labels corresponding to each sample in the `data`.
            verbose (bool, optional): ff `True`, prints detailed information. (Default: `False`).

        Returns:
            int: the number of correctly predicted labels.
        """
        total_cnt = len(Y)
        Y_hat = self.predict(X)
        correct = 0 
        for i in range(total_cnt):
            if Y_hat[i] == Y[i]:
                correct += 1

        ratio = correct/total_cnt
 
        if verbose:
            print(f"[EVAL] Accuracy: {ratio*100:.2f}%, Error rate: {(1-ratio)*100:.2f}%.")

        return ratio