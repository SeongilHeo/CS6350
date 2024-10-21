import re
import random

def load_data_desc(filename):
    """
    Load dataset description from a text file.

    Args:
        filename (str): path to the file containing the dataset description.

    Returns:
        tuple: A tuple containing:
            - labels (list): possible labels.
            - attributes (dict): 'keys': attribute names, 'values': possible attribute values (if numeric is None).
            - categorical_attributes (set): attribute names that are categorical.
            - numerical_attributes (set): attribute names that are numerical.
            - missing_attributes (set): attribute names that are contained missing value(?).
            - columns (dict): 'keys': attribute names, 'keys': column index.
    
    Example file format:
    ```
    | label values

    0,1

    | attributes

    - x1: 0,1.

    - x2: 0,1.

    - x3: 0,1.

    - x4: 0,1.

    - x5: 

    | columns
    x1,x2,x3,x4
    ```
    """
    labels = []
    attributes = {}
    categorical_attributes = set()
    numerical_attributes = set()
    missing_attributes = set()
    columns = []

    # default value of section
    section = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            # set section
            if line.startswith('|'):
                section = line[1:].strip()
                continue
            
            # parse labels
            if section == 'label values':
                labels = [label.strip() for label in line.split(',')]
            # parse attributes
            elif section == 'attributes':
                match = re.findall(r"[\s]*([\w.]+)[\s]*:[\s]*([\w\s\-,.?\(\)&]+)\.", line)[0]
                key = match[0]
                values = re.findall(r"([\w\(\)?\-&.]+)", match[1])
                if values:
                    categorical_attributes.add(key)
                    if 'unknown' in values:
                        missing_attributes.add(key)
                        # values.remove('?')
                else:
                    numerical_attributes.add(key)
                attributes[key] = values
            # parse columns
            elif section == 'columns':
                columns = {col.strip(): idx for idx, col in enumerate(line.split(','))}

    return labels, attributes, categorical_attributes, numerical_attributes, missing_attributes, columns

def load_data(filename):
    """
    Load dataset from a CSV file.

    Args:
        filename (str): path to the CSV file to be loaded.

    Returns:
        tuple: data (list of lists), labels (list)
    """
    data=[]
    labels=[]
    try:
        with open(filename, 'r') as f:
            for line in f:
                sample = line.strip().split(',')
                data.append(sample[:-1])
                labels.append(sample[-1])
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File '{filename}' not found.")
    except Exception as e:
        raise ValueError(f"[Error] Failed file read: {e}")

    return data, labels

def preprosess_numeric(X, numerical_attributes, columns):
    """
    Prprocess for numerical attribute from 'str' to 'float'.

    Args:
        X: data samples.
        numerical_attributes: numerical attribute naems.
        columns: 'keys': attribute names, 'values': column index.

    Returns:
        list of list: preprocessed data samples
    """
    for sample in X:
        for attribute in numerical_attributes:
            column_idx = columns[attribute]
            sample[column_idx] = float(sample[column_idx])
    return X

def preprosess_miss(X, idx):
    """
    Prprocess for missing values from 'unknown' to the maximum value in the dataset.

    Args:
        X: data samples.
        idx: index having unknown data.

    Returns:
        list of list: preprocessed data samples
    """
    original_X = [sample[idx] for sample in X]

    # count attribute values
    counter = count_label(original_X)

    # remove unknown's count
    if counter.get("unknown"): counter.pop("unknown")

    # sort by count
    substitue = sorted(list(counter.items()), key=lambda x:x[1], reverse=True)[0][0]

    # replace to maximum attribute value
    for sample in X:
        if sample[idx] == "unknown":
            sample[idx] = substitue
    return X

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

def bootstrap_sample(X,Y, num_sample=None, random_state=None, replace = True):
    """
    Bootstrap from sampel X, Y.

    Args:
        X (list): data samples.
        Y (list): labels corresponding 'X.
        num_sample (float): number of smapling data.
        replace (bool): option with or without replacement.

    Returns:
        tuple: boostrap_X, boostrap_Y (with replacement)
    """
    if random_state is not None:
        random.seed(random_state)
    
    num_data = len(Y)

    # with replacement
    if replace:
        indices = random.choices(range(num_data), k=num_sample)
    # without replacement
    else:
        indices = random.sample(range(num_data), k=num_sample)

    X_bootstrap = [X[i] for i in indices]
    Y_bootstrap = [Y[i] for i in indices]
    
    return X_bootstrap, Y_bootstrap

def covert_labels(Y,afters=[1,-1]):
    """
    Preprosesss for Adaboost algorithm.

    Args:
        Y (list): labels of dataset.
        after (list): candidates of new labels

    Returns:
        list: converted labels
    """
    befores=set(Y)
    table = {before:after for before,after in zip(befores,afters)}
    return [table[y] for y in Y]

def args_num(num):
    if '-' in num:
        return tuple(map(int,num.split("-")))
    else:
        return (int(num),int(num)+1)

def calculate_average(Y):
    return sum(Y)/len(Y)

# def calculate_bias(Y,Y_hat_avg):
#     return [(Y[i]-Y_hat_avg[i])**2 for i in range(len(Y))]
    
# def calculate_var(Y,Y_hat):
#     m = len(Y)
#     return [(Y[i]-Y_hat[i])**2 for i in range(m)]
    
# def calculate_bias_var(Y,Y_hat):
#     bias = calculate_bias(Y,Y_hat)
#     var = calculate_average(bias)
#     return bias, var

