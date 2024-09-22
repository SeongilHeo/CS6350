def load_data_desc(filename):
    labels = []
    attributes = {}
    numerical_attributes = []
    columns = []

    section = None
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue
                
            if line.startswith('|'):
                section = line[1:].strip()
                continue

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

    return labels, attributes, numerical_attributes, columns

def load_data(filename):
    data=[]
    labels=[]

    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            data.append(row[:-1])
            labels.append(row[-1])
    return data, labels

def preprosseing(data,numerical_attributes, columns):
    for row in data:
        for a in numerical_attributes:
            idx = columns[a]
            row[idx] = float(row[idx])
    return data

def preprosseing(data,numerical_attributes, columns):
    for row in data:
        for a in numerical_attributes:
            idx = columns[a]
            row[idx] = float(row[idx])
    return data

def missing(data,idx):
    pre_data = [row[idx] for row in data]
    counter = count_label(pre_data)
    if counter.get("unknown"): counter.pop("unknown")

    substitue = sorted(list(counter.items()), key=lambda x:x[1])[-1][0]
    for d in data:
        if d[idx] == "unknown":
            d[idx] = substitue
    return data

def count_label(labels):
    label_counter = {}

    for label in labels:
        if label in label_counter:
            label_counter[label] += 1
        else:
            label_counter[label] = 1
    return label_counter

