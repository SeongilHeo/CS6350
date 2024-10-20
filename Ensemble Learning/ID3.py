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
from math import log2

def count_label(labels):
    """
    Count the occurrences of each label in the given list of labels.

    Args:
        labels (list): labels.

    Returns:
        dict: 'keys': label, 'values': count.
    """
    label_counter = {}

    for label in labels:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1
    return label_counter

def entropy(labels):
    """
    Calculate the entropy of a list of labels for measureing the impurity of the label set

    H(X) = - Σ p(x) * log2(p(x))

    Args:
        labels (list): labels.

    Returns:
        float: entropy of the label distribution.
               between 0 and 1. (0, all labels are the same)

    """
    total_cnt = len(labels)
    label_counter=count_label(labels)

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

def gini_index(labels):
    """
    Calculate the Gini index of a list of labels for measureing the impurity of the label set
    
    GI(X) = 1 - Σ p(x)^2

    Args:
        labels (list): labels.

    Returns:
        float: gini index of the label distribution.
               between 0 and 1. (0, all labels are the same)
    """
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
        train(data, labels, weights, attributes, columns, numerical_attributes=[])
        _build_tree(data, labels, weights, attributes, depth)
        _split_data(data, labels, weights, attribute)
        _best_split(data, labels, weights, attributes)
        _most_common_label(labels)
        _convert_numerical_to_binary(data, attribute, threshold)
        predict(data)
        evaluate(data, labels, verbose=False)
        _classify(row, node)
        visualization()
        _traverse(self, node, depth)
    """
    def __init__(self, max_depth=None, criterion='information_gain'):
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

    # for train -------------------------------------------------------------------------------------------------------------------
    def train(self, data, labels, weights, attributes, columns, numerical_attributes=[]):
        """
        Train the ID3 decision tree on the given dataset.

        Args:
            data (list of lists): data samples.
            labels (list): labels corresponding to each sample in the `data`.
            weights (list): weights corresponding to each sample in the `data`.
            attributes (dict): 'keys': attribute names, 'values': possible values, if numerical attribute maps `None` value 
            columns (dict): 'keys': attribute names, 'values': column index in the dataset.
            numerical_attributes (list, optional): numerical attributes. 

        Returns:
            None:
        """
        self.columns = columns
        self.attributes = attributes
        self.numerical_attributes = numerical_attributes
        
        # build tree using data
        self.root = self._build_tree(data, labels, weights, attributes=list(self.attributes.keys()), depth=0)

    def _build_tree(self, data, labels, weights, attributes, depth):
        """    
        Recursively build the decision tree.

            Args:
                data (list of lists): data samples.
                labels (list): labels corresponding to each sample in the `data`.
                weights (list): weights corresponding to each sample in the `data`.
                attributes (list): attribute names available for splitting at the current.
                depth (int): current depth of the tree.

            Returns:
                Node: the root node of the subtree built at the current recursion level.
        """
        # base case (all examples have same label)
        if len(set(labels)) == 1:
            return Node(label=labels[0])

        # base case (arrive max_depth or attributes empty)
        if depth == self.max_depth or not attributes:
            return Node(label=self._most_common_label(labels))

        # select best split attribute
        best_attribute, best_threshold = self._best_split(data, labels, weights, attributes)

        # remove selected best attribute
        child_attribute=attributes[:]
        child_attribute.remove(best_attribute)

        # branch children
        node = Node(attribute=best_attribute, threshold=best_threshold)
        
        # for numerical attribute
        if best_threshold is not None:
            data = self._convert_numerical_to_binary(data, best_attribute, best_threshold)

        subsets = self._split_data(data, labels, weights, best_attribute)

        # recursion or stop
        for attribute_value, (subset_data, subset_labels, subset_weights) in subsets.items():
            if len(subset_labels) == 0:
                child_node = Node(label=self._most_common_label(labels))
            else:
                child_node = self._build_tree(subset_data, subset_labels, subset_weights, child_attribute, depth + 1)
            node.children[attribute_value] = child_node

        return node 

    def _split_data(self, data, labels, weights, attribute):
        """
        Split the data based on the given attribute.

        Args:
            data (list of list): data samples.
            labels (list): labels corresponding to each sample in the `data`.
            weights (list): weights corresponding to each sample in the `data`.
            attribute (str): selected attribute name to split the `data`.

        Returns:
            dict: 'keys': attribute values (0 and 1 for numerical attributes).
                  'values': tuple. (list of subset data, list of subset labels, list of subset weights).
        """
        # numerical attribute
        if attribute in self.numerical_attributes:
            splits = {attribute_value: ([],[],[]) for attribute_value in [0,1]}
        # categorical or binary attribute
        else:
            splits = {attribute_value: ([],[],[]) for attribute_value in self.attributes[attribute]}

        attribute_index = self.columns[attribute]

        # split data        
        for i in range(len(data)):
            attribute_value = data[i][attribute_index]
            splits[attribute_value][0].append(data[i])
            splits[attribute_value][1].append(labels[i])
            splits[attribute_value][2].append(weights[i])

        return splits

    def _best_split(self, data, labels, weights, attributes):
        """
        Find the best attribute and threshold for splitting the dataset.

        Args:
            data (list of lists): data samples
            labels (list): lables corresponding to each sample in the `data`.
            weights (list): weights corresponding to each sample in the `data`.
            attributes (list): attribute names available for splitting at the current.
        
        Returns:
            tuple: best attribute name, best threshold (or `None`)
        """
        max_gain = -float('inf')
        best_attribute = None
        best_threshold = None

        base_criterion_value = self.criterion_func(labels)

        # compare attributes
        for attribute in self.attributes.keys():
            # preselected attributes
            if attribute not in attributes:
                continue

            # numerical attribute
            if attribute in self.numerical_attributes:
                numerical_data = [sample[self.columns[attribute]] for sample in data]
                threshold = median(numerical_data)
                binary_data = self._convert_numerical_to_binary(data, attribute, threshold)
                splits = self._split_data(binary_data, labels, weights, attribute)
            # categorical or binary attribute
            else:
                splits = self._split_data(data, labels, weights, attribute)

            # calcualte information gain with criterion value
            weighted_avg = 0.0
            for subset_data, subset_labels, subset_weights in splits.values():
                if not subset_labels: continue
                weighted_avg += sum(subset_weights) * self.criterion_func(subset_labels)
            gain = base_criterion_value - weighted_avg
    
            # maximum information gain
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
                best_threshold =  threshold if attribute in self.numerical_attributes else None

        return best_attribute, best_threshold
    
    def _most_common_label(self, labels):

        """
        Determine the most common label for leaf node.

        Args:
            labels (list): labels.

        Returns:
            any: label that appears most frequently in the `labels`.
        """
        label_counts=count_label(labels)
        return max(label_counts, key=label_counts.get)
    
    # for numerical attribute
    def _convert_numerical_to_binary(self, data, attribute, threshold):
        """
        Convert numerical attribute values into binary form based on a threshold.

        Args:
            data (list of lists): data samples.
            attribute (str): numerical attribute name.
            threshold (float or int): threshold value used to split.

        Returns:
            list of lists: new dataset replaced by 0 or 1.
        """
        binary_data = []
        attribute_idx = self.columns[attribute]
        for sample in data:
            binary_sample = sample[:]
            if sample[attribute_idx] < threshold:
                binary_sample[attribute_idx] = 0
            else:
                binary_sample[attribute_idx] = 1
            binary_data.append(binary_sample)
        return binary_data

    # for test---------------------------------------------------------------------------------------------------------------------
    def predict(self, data):
        """
        Predict the labels for the data.
        
        Args: 
            data (list of list): data samples.

        Results:
            list: predicted labels for the `data`.
        """
        predictions = []

        for sample in data:
            predictions.append(self._classify(sample, self.root))

        return predictions
    
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