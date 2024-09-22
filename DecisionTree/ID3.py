
def median(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]
    
# Purity measure Functions----------------------------------------------
from math import log2

def count_label(labels):
    label_counter = {}

    for label in labels:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1
    return label_counter

def entropy(labels):
    total_cnt = len(labels)
    label_counter=count_label(labels)

    ent = 0.0
    for cnt in label_counter.values():
        p = cnt / total_cnt
        ent -= p * (log2(p))
    return ent

def majority_error(labels):
    total_cnt = len(labels)
    label_counter=count_label(labels)
    
    most_common_cnt = max(label_counter.values())
    return 1 - (most_common_cnt / total_cnt)

def gini_index(labels):
    total_cnt = len(labels)
    label_counter=count_label(labels)
    
    gini = 1.0
    for cnt in label_counter.values():
        p = cnt / total_cnt
        gini -= p ** 2
    return gini
# ----------------------------------------------------------------------

class Node:
    def __init__(self, attribute=None, threshold=None, children=None, label=None):
        self.attribute = attribute 
        self.threshold = threshold 
        self.children = children or {}
        self.label = label

class ID3:
    def __init__(self, max_depth=None, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion_func = { 'information_gain': entropy, 
                                'majority_error': majority_error,
                                'gini_index': gini_index }[criterion]
        self.root = None
        self.attributes = None
        self.numerical_attributes = None
        self.columns = None

    # for train --------------------------------------------------------
    def train(self, data, labels, attributes, columns, numerical_attributes=[]):
        self.columns = columns
        self.attributes = attributes
        self.numerical_attributes = numerical_attributes

        # adjust max_depth by attributes cnt
        self.max_depth = min(len(self.attributes), self.max_depth) if self.max_depth else len(self.attributes)

        # build tree using data
        self.root = self._build_tree(data, labels, depth=0)

    def _build_tree(self, data, labels, depth):
        # base case (all examples have same label)
        if len(set(labels)) == 1:
            return Node(label=labels[0])

        # base case (arrive max_depth or attributes empty)
        if depth == self.max_depth:
            return Node(label=self._most_common_label(labels))

        # select best split attribute
        attribute, threshold = self._best_split(data, labels)

        # branch children
        node = Node(attribute=attribute, threshold=threshold)
        
        if threshold is not None:
            data = self._convert_numerical_to_binary(data, attribute, threshold)

        subsets = self._split_data(data, labels, attribute)

        # recursion or stop
        for attribute_value, (subset_data, subset_labels) in subsets.items():
            if len(subset_labels) == 0:
                child_node = Node(label=self._most_common_label(labels))
            else:
                child_node = self._build_tree(subset_data, subset_labels, depth + 1)
            node.children[attribute_value] = child_node

        return node 

    def _split_data(self, data, labels, attribute):
        if attribute in self.numerical_attributes:
            splits = {attribute_value: ([],[]) for attribute_value in [0,1]}
        else:
            splits = {attribute_value: ([],[]) for attribute_value in self.attributes[attribute]}
        attribute_index = self.columns[attribute]
        
        for i in range(len(data)):
            attribute_value = data[i][attribute_index]
            splits[attribute_value][0].append(data[i])
            splits[attribute_value][1].append(labels[i])

        return splits

    def _best_split(self, data, labels):
        max_gain = -float('inf')
        best_attribute = None
        best_threshold = None

        total_samples = len(labels)
        base_criterion_value = self.criterion_func(labels)

        for attribute in self.attributes.keys():
            if attribute in self.numerical_attributes:
                numerical_data = [sample[self.columns[attribute]] for sample in data]
                threshold = median(numerical_data)
                binary_data = self._convert_numerical_to_binary(data, attribute, threshold)
                splits = self._split_data(binary_data, labels, attribute)
            else:
                splits = self._split_data(data, labels, attribute)

            weighted_avg = 0.0

            for _, subset_labels in splits.values():
                if not subset_labels: continue
                weight = len(subset_labels) / total_samples
                weighted_avg += weight * self.criterion_func(subset_labels)

            gain = base_criterion_value - weighted_avg
            if gain > max_gain:
                max_gain = gain
                best_attribute = attribute
                if attribute in self.numerical_attributes:
                    best_threshold = threshold
                else:
                    best_threshold = None

        return best_attribute, best_threshold
    
    def _most_common_label(self, labels):
        label_counts=count_label(labels)
        return max(label_counts, key=label_counts.get)
    # ------------------------------------------------------------------

    # for test----------------------------------------------------------
    def predict(self, data):
        predictions = []
        for row in data:
            predictions.append(self._classify(row, self.root))
        return predictions
    
    def _classify(self, row, node):
        if node.label is not None:
            return node.label
        attribute_value = row[self.columns[node.attribute]]

        if node.attribute in self.numerical_attributes:
            attribute_value = 0 if attribute_value < node.threshold else 1

        child_node = node.children[attribute_value]

        return self._classify(row, child_node)

    def evaluate(self, data, labels, verbose=False):
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
    
    # for binary--------------------------------------------------------
    def _convert_numerical_to_binary(self, data, attribute, threshold):
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