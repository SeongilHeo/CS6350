import random
import numpy as np

def load_data(filename):
    """
    Load dataset from a CSV file.

    Args:
        filename (str): path to the CSV file to be loaded.

    Returns:
        tuple: X (np.array), y (np.array)
    """
    try:
        data = np.loadtxt(filename, delimiter=",")
        X = data[:, :-1]
        y = data[:, -1]
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] File '{filename}' not found.")
    except Exception as e:
        raise ValueError(f"[Error] Failed file read: {e}")

    return X, y


# def args_num(num):
#     if '-' in num:
#         return tuple(map(int,num.split("-")))
#     else:
#         return (int(num),int(num)+1)

# def calculate_average(Y):
#     return sum(Y)/len(Y)