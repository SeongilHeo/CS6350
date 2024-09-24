def load_data_desc(filename):
    """
    Load dataset description from a text file.

    Args:
        filename (str): path to the file containing the dataset description.

    Returns:
        tuple: A tuple containing:
            - labels (list): possible labels.
            - attributes (dict): 'keys': attribute names, 'values': possible attribute values (if numeric is None).
            - numerical_attributes (list): attribute names that are numerical.
            - columns (dict): 'keys': attribute names, 'keys': column index.
    
    Example file format:
    ```
    | label values

    0,1

    | attributes

    x1: 0,1.
    x2: 0,1.
    x3: 0,1.
    x4: 0,1.

    | columns
    x1,x2,x3,x4
    ```
    """
    labels = []
    attributes = {}
    numerical_attributes = []
    columns = []

    # default value of section
    section = None
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                
                if not line:
                    continue
                # set section
                if line.startswith('|'):
                    section = line[1:].strip()
                    continue
                # parse attributes
                if section == 'label values':
                    labels = [label.strip() for label in line.split(',')]
                elif section == 'attributes':
                    key, values = line.split(':')
                    if "(" in values:
                        numerical_attributes.append(key.strip())
                        attributes[key.strip()] = None
                        continue
                    attributes[key.strip()] = [v.strip() for v in values[:-1].split(',')]
                elif section == 'columns':
                    columns = {col.strip(): idx for idx, col in enumerate(line.split(','))}
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File '{filename}' not found.")
    except Exception as e:
        raise ValueError(f"[Error] Failed file read: {e}")
    
    return labels, attributes, numerical_attributes, columns

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
        raise FileNotFoundError(f"File '{filename}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

    return data, labels

def preprosess_numeric(data, numerical_attributes, columns):
    """
    Prprocess for numerical attribute from 'str' to 'float'.

    Args:
        data: data samples.
        numerical_attributes: numerical attribute naems.
        columns: 'keys': attribute names, 'values': column index.

    Returns:
        list of list: preprocessed data samples
    """
    for sample in data:
        for attribute in numerical_attributes:
            column_idx = columns[attribute]
            sample[column_idx] = float(sample[column_idx])
    return data

def preprosess_miss(data, idx):
    """
    Prprocess for missing values from 'unknown' to the maximum value in the dataset.

    Args:
        data: data samples.
        idx: index having unknown data.

    Returns:
        list of list: preprocessed data samples
    """
    original_data = [sample[idx] for sample in data]

    # count attribute values
    counter = count_label(original_data)

    # remove unknown's count
    if counter.get("unknown"): counter.pop("unknown")

    # sort by count
    substitue = sorted(list(counter.items()), key=lambda x:x[1], reverse=True)[0][0]

    # replace to maximum attribute value
    for sample in data:
        if sample[idx] == "unknown":
            sample[idx] = substitue
    return data

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

