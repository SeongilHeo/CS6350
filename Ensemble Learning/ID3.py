from math import log2
import random 

def median(values):
    """
    Calculate the median of a list of values.

    Args:
        values (list): numerical values.

    Returns:
        float or int: median of `values`.
    """
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]
    
# Purity measure Functions---------------------------------------------------------------------------------------------------------

def count_label(labels, weights=None):
    """
    Count the occurrences of each label in the given list of labels.
    OR
    Calcualte the sum of samples weight occurrences of each label in the given list of labels.

    Args:
        labels (list): labels.
        weights (list, optional): weight.

    Returns:
        dict: 'keys': label, 'values': count.
    """
    if weights:
        weights_sum = {}
        for idx, label in enumerate(labels):
            if label in weights_sum:
                weights_sum[label] += weights[idx]
            else:
                weights_sum[label] = weights[idx]
        return weights_sum
    else:
        label_counter = {}
        for label in labels:
            if label in label_counter:
                label_counter[label] += 1
            else:
                label_counter[label] = 1
        return label_counter

def entropy(labels, weights = None):
    """
    Calculate the entropy of a list of labels for measureing the impurity of the label set

    H(X) = - Σ p(x) * log2(p(x))

    Args:
        labels (list): labels.
        weights (options): weights.

    Returns:
        float: entropy of the label distribution.
               between 0 and 1. (0, all labels are the same)

    """
    if weights:
        weights_sum = count_label(labels, weights)
        total_weight =  sum(weights)

        ent = 0.0
        for w in weights_sum.values():
            p = w / total_weight
            ent -= p * (log2(p) + 1e-9)
    else:
        label_counter=count_label(labels)
        total_cnt = len(labels)

        ent = 0.0
        for cnt in label_counter.values():
            p = cnt / total_cnt
            ent -= p * (log2(p))
    return ent

def majority_error(labels):
    """
    Calculate the majority error of a list of labels for measureing the impurity of the label set

    ME(X) = 1 - (count of most common label / total count)

    Args:
        labels (list): labels.

    Returns:
        float: majority error of the label distribution.
               between 0 and 1. (0, all labels are the same)
    """
    total_cnt = len(labels)
    label_counter=count_label(labels)
    
    most_common_cnt = max(label_counter.values())
    return 1 - (most_common_cnt / total_cnt)

def gini_index(labels,weights=None):
    """
    Calculate the Gini index of a list of labels for measureing the impurity of the label set
    
    GI(X) = 1 - Σ p(x)^2

    Args:
        labels (list): labels.
        weights (list): weights.

    Returns:
        float: gini index of the label distribution.
               between 0 and 1. (0, all labels are the same)
    """
    if weights:
        weights_sum = count_label(labels, weights)
        total_weight =  sum(weights)

        gini = 1.0
        for w in weights_sum.values():
            p = w / total_weight
            gini -= p ** 2
    else:
        total_cnt = len(labels)
        label_counter=count_label(labels)
        
        gini = 1.0
        for cnt in label_counter.values():
            p = cnt / total_cnt
            gini -= p ** 2
    return gini

# Node Class------------------------------------------------------------------------------------------------------------------------
class Node:
    """
    A class representing a node in a decision tree.

    Attributes:
        attribute (str, optional): attribute used to split the data.
        threshold (float or int, optional): threshold value for the numerical attribute. 
        children (dict, optional): child nodes.
        label (any, optional): assigned label to this node. (only leaf node)
    """
    def __init__(self, attribute=None, threshold=None, children=None, label=None):
        self.attribute = attribute 
        self.threshold = threshold 
        self.children = children or {}
        self.label = label

# ID3 Class------------------------------------------------------------------------------------------------------------------------
class ID3:
    """
    A class that implements the ID3 decision tree algorithm.

    Attributes:
        max_depth (int, optional)
        criterion_func (function)
        root (Node, optional)
        attributes (dict, optional)
        numerical_attributes (list, optional)

    Methods:
        train(X, Y, attributes, columns, numerical_attributes=[], weights=None)
        _build_tree(X, y, attributes, depth, weights=None)
        _split_data(X, Y, attribute, weights)
        _best_split(X, Y, attributes, weights)
        _most_common_label(Y)
        _convert_numerical_to_binary(X, attribute, threshold)
        predict(X)
        evaluate(data, labels, verbose=False)
        _classify(sample, node)
        visualization()
        _traverse(self, node, depth)
    """
    def __init__(self, max_depth=None, criterion='information_gain', num_attributes = None):
        """
        Constructor.

        Args:
            max_depth (int, optional): maximum depth the tree.
            criterion (str, optional): criterion used to evaluate the inpurity of data.
                                       ['information_gain', 'majority_error', 'gini_index'].
                                       (Default:'information_gain').
        """
        self.max_depth = max_depth
        self.criterion_func = { 'information_gain': entropy, 
                                'majority_error': majority_error,
                                'gini_index': gini_index }[criterion]
        self.root = None
        self.attributes = None
        self.numerical_attributes = None
        self.columns = None
        self.weights = None
        self.num_attributes = num_attributes

    # for train -------------------------------------------------------------------------------------------------------------------
    def train(self, X, Y, attributes, columns, numerical_attributes=[], weights=None):
        """
        Train the ID3 decision tree on the given dataset.

        Args:
            X (list of lists): data samples.
            Y (list): labels corresponding to each sample in the `X`.
            attributes (dict): 'keys': attribute names, 'values': possible values, if       numerical attribute maps `None` value 
            columns (dict): 'keys': attribute names, 'values': column index in the dataset.
            numerical_attributes (list, optional): numerical attributes. 
            weights (list): weights corresponding to each sample in the `X`.

        Returns:
            None:
        """
        self.attributes = attributes
        self.columns = columns
        self.numerical_attributes = numerical_attributes
        self.weights = weights

        # build tree using data
        self.root = self._build_tree(X, Y, attributes=list(self.attributes.keys()), depth=0, weights = weights)

    def _build_tree(self, X, Y, attributes, depth, weights=None):
        """    
        Recursively build the decision tree.

            Args:
                X (list of lists): data samples.
                Y (list): labels corresponding to each sample in the `X`.
                attributes (list): attribute names available for splitting at the current.
                depth (int): current depth of the tree.
                weights (list): weights corresponding to each sample in the `X`.

            Returns:
                Node: the root node of the subtree built at the current recursion level.
        """
        # base case (all examples have same label)
        if len(set(Y)) == 1:
            return Node(label=Y[0])

        # base case (arrive max_depth or attributes empty)
        if depth == self.max_depth or not attributes:
            return Node(label=self._most_common_label(Y))

        if self.num_attributes:
            bootstrap_attribute = random.sample(attributes, min(self.num_attributes, len(attributes)))

            # select best split attribute
            best_attribute, best_threshold = self._best_split(X, Y, bootstrap_attribute, weights)
        else:
            # select best split attribute
            best_attribute, best_threshold = self._best_split(X, Y, attributes, weights)

        # remove selected best attribute
        child_attribute=attributes[:]
        child_attribute.remove(best_attribute)

        # branch children
        node = Node(attribute=best_attribute, threshold=best_threshold)
        
        # for numerical attribute
        if best_threshold is not None:
            X = self._convert_numerical_to_binary(X, best_attribute, best_threshold)

        subsets = self._split_data(X, Y, best_attribute, weights)

        # recursion or stop
        if weights:
            for attribute_value, (subset_X, subset_Y, subset_weights) in subsets.items():
                if len(subset_Y) == 0:
                    child_node = Node(label=self._most_common_label(Y))
                else:
                    child_node = self._build_tree(subset_X, subset_Y, child_attribute, depth + 1, weights=subset_weights)
                node.children[attribute_value] = child_node
        else:
            for attribute_value, (subset_X, subset_Y) in subsets.items():
                if len(subset_Y) == 0:
                    child_node = Node(label=self._most_common_label(Y))
                else:
                    child_node = self._build_tree(subset_X, subset_Y, child_attribute, depth + 1)
                node.children[attribute_value] = child_node

        return node 

    def _split_data(self, X, Y, attribute, weights):
        """
        Split the data based on the given attribute.

        Args:
            X (list of list): data samples.
            Y (list): labels corresponding to each sample in the `X`.
            attribute (str): selected attribute name to split the `X`.
            weights (list): weights corresponding to each sample in the `X`.

        Returns:
            dict: 'keys': attribute values (0 and 1 for numerical attributes).
                  'values': tuple. (list of subset X, list of subset Y, list of subset weights) or (list of subset X, list of subset Y).
        """
        attribute_index = self.columns[attribute]

        if weights:
            # numerical attribute
            if attribute in self.numerical_attributes:
                splits = {attribute_value: ([],[],[]) for attribute_value in [0,1]}
            # categorical or binary attribute
            else:
                splits = {attribute_value: ([],[],[]) for attribute_value in self.attributes[attribute]}

            # split data        
            for i in range(len(X)):
                attribute_value = X[i][attribute_index]
                splits[attribute_value][0].append(X[i])
                splits[attribute_value][1].append(Y[i])
                splits[attribute_value][2].append(weights[i])
        else:
            # numerical attribute
            if attribute in self.numerical_attributes:
                splits = {attribute_value: ([],[]) for attribute_value in [0,1]}
            # categorical or binary attribute
            else:
                splits = {attribute_value: ([],[]) for attribute_value in self.attributes[attribute]}

            # split data        
            for i in range(len(X)):
                attribute_value = X[i][attribute_index]
                splits[attribute_value][0].append(X[i])
                splits[attribute_value][1].append(Y[i])

        return splits

    def _best_split(self, X, Y, attributes, weights):
        """
        Find the best attribute and threshold for splitting the dataset.

        Args:
            X (list of lists): data samples
            Y (list): lables corresponding to each sample in the `X`.
            attributes (list): attribute names available for splitting at the current.
            weights (list): weights corresponding to each sample in the `X`.
        
        Returns:
            tuple: best attribute name, best threshold (or `None`)
        """
        max_gain = -float('inf')
        best_attribute = None
        best_threshold = None

        total_samples = len(Y)
        base_criterion_value = self.criterion_func(Y,weights)

        # compare attributes
        for attribute in self.attributes.keys():
            # preselected attributes
            if attribute not in attributes:
                continue

            # numerical attribute
            if attribute in self.numerical_attributes:
                numerical_X = [sample[self.columns[attribute]] for sample in X]
                threshold = median(numerical_X)
                binary_X = self._convert_numerical_to_binary(X, attribute, threshold)
                splits = self._split_data(binary_X, Y, attribute, weights)
            # categorical or binary attribute
            else:
                splits = self._split_data(X, Y, attribute, weights)

            # calcualte information gain with criterion value
            weighted_avg = 0.0
            # for adaboost
            if weights:
                total_weights=sum(weights)
                for subset_X, subset_Y, subset_weights in splits.values():
                    if not subset_Y: continue
                    weighted_avg += (sum(subset_weights)/ total_weights) * self.criterion_func(subset_Y,subset_weights)
            # base
            else:
                for subset_X, subset_Y in splits.values():
                    if not subset_Y: continue
                    weighted_avg += (len(subset_Y)/ total_samples) * self.criterion_func(subset_Y)

            gain = base_criterion_value - weighted_avg
        
            # maximum information gain
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
                best_threshold =  threshold if attribute in self.numerical_attributes else None

        return best_attribute, best_threshold
    
    def _most_common_label(self, Y):

        """
        Determine the most common label for leaf node.

        Args:
            Y (list): labels.

        Returns:
            any: label that appears most frequently in the `Y`.
        """
        label_counts=count_label(Y)
        return max(label_counts, key=label_counts.get)
    
    # for numerical attribute
    def _convert_numerical_to_binary(self, X, attribute, threshold):
        """
        Convert numerical attribute values into binary form based on a threshold.

        Args:
            X (list of lists): data samples.
            attribute (str): numerical attribute name.
            threshold (float or int): threshold value used to split.

        Returns:
            list of lists: new dataset replaced by 0 or 1.
        """
        binary_X = []
        attribute_idx = self.columns[attribute]
        for sample in X:
            binary_sample = sample[:]
            if sample[attribute_idx] < threshold:
                binary_sample[attribute_idx] = 0
            else:
                binary_sample[attribute_idx] = 1
            binary_X.append(binary_sample)
        return binary_X

    # for test---------------------------------------------------------------------------------------------------------------------
    def predict(self, X):
        """
        Predict the labels for the data.
        
        Args: 
            X (list of list): data samples.

        Results:
            list: predicted labels for the `data`.
        """
        Y_hat = []
        for sample in X:
            Y_hat.append(self._classify(sample, self.root))

        return Y_hat
    
    def _classify(self, sample, node):
        """
        Recursively classify a single data sample by traversing the decision tree.

        Args:
            sample (list): single sample.
            node (Node): current node of the tree during traversal.

        Returns:
            any: predicted label for the `sample`.
        """
        # current node is leaf node
        if node.label is not None:
            return node.label

        attribute_value = sample[self.columns[node.attribute]]

        # current node is numerical attribute
        if node.attribute in self.numerical_attributes:
            attribute_value = 0 if attribute_value < node.threshold else 1

        # select next level child
        child_node = node.children[attribute_value]

        return self._classify(sample, child_node)

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
        total_cnt = len(Y)
        Y_hat = self.predict(X)

        result = 0 
        for i in range(total_cnt):
            if Y_hat[i] == Y[i]:
                result += 1
        
        ratio = result/total_cnt
        
        if verbose:
            print(f"[Info] Out of {total_cnt} data points, {result} match. Accuracy: {ratio*100}%, Error rate: {(1-ratio)*100}%.")

        return result
    
    # for visualization -----------------------------------------------------------------------------------------------------------
    def visualization(self):
        """
        Visualization of the decision tree.

        Args:

        Returns:
            None: 
        """
        if not self.root:
            print("[Info] not trained tree")
            return
        
        print("="*70)
        self._traverse(self.root,0)
        print("="*70)

    def _traverse(self, node, depth):
        """
        DFS traverse the decision tree for visualization..

        Args:
            node: current node.
            depth: current depth.

        Returns:
            None: This method prints the tree structure.
        """
        # not leaf node
        indent = "  " * depth
        if node.label is None: 
            if node.threshold is not None:
                print(f"{indent}Node: {node.attribute} < {node.threshold}")
            else:
                print(f"{indent}Node: {node.attribute}")

            # next level
            for attribute_value, child in node.children.items():
                print(f"{indent}  Branch: {attribute_value}")
                self._traverse(child,depth+2)
        # leaf node
        else: 
            print(f"{indent}Leaf: Label = {node.label}")