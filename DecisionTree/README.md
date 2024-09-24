# Homework 1 - Decision Tree
CS 5350/6350: Machine Learning Fall 2024  

- Handed out: 3 Sep, 2024  
- Due: 11:59pm, ~~20 Sep, 2024~~ -> 24 Sep, 2024
- Submission: ~~21 Sep, 2024~~ -> 24 Sep, 2024

## Dataset
- Car Dataset: UCI repository(https://archive.ics.uci.edu/ml/datasets/car+evaluation).
- Bank Dataset: UCI repository(https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).
## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SeongilHeo/CS6350.git
   cd DecisionTree
   ```

2. Place Dataset in the same folder.  
`data-desc.txt` file in the same format as the car dataset is required. Place the data-desc.txt file into the Data/bank folder for the bank dataset, or simply place it in the current directory.    
   ```
    .
    ├── DecisionTree
    │   ├── Data
    │   │   ├── bank
    │   │   │   ├── data-desc.txt
    │   │   │   ├── test.csv
    │   │   │   └── train.csv
    │   │   ├── bool
    │   │   │   ├── data-desc.txt
    │   │   │   └── train.csv
    │   │   ├── car
    │   │   │   ├── data-desc.txt
    │   │   │   ├── test.csv
    │   │   │   └── train.csv
    │   │   └── golf
    │   │       ├── data-desc.txt
    │   │       └── train.csv
    │   ├── ID3.py
    │   ├── README.md
    │   ├── data-desc.txt
    │   ├── run.py
    │   ├── run.sh
    │   └── utils.py
    └── README.md
    ```

3. Make the `run.sh` script executable:
   ```bash
   chmod +x run.sh
   ```

## Usage

The `run.sh` script accepts several command-line options to control the behavior of the `run.py` Python script.

### General Command Structure

```bash
./run.sh [options]
```

### Options

| Option                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--help`               | Show this help message and exit.                                             |
| `--data DATA`          | Choose Dataset: `car`, `bank`. (Default: `car`).                             |
| `--dir DIR`            | Directory of the data folder. (Default:`./Data`. Example: `/path/to/Data`). |
| `-M`, `--miss`         | Enable this flag to handle "unknown" values in the data.                     |
| `-T`, `--tree`         | Enable this flag to visualize the decision tree.                             |
| `-D DEPTH`, `--depth DEPTH` | Set the max depth of the ID3 decision tree. (Default: 1 to Max).           |
| `-C CRITERION`, `--criterion CRITERION` | Set the criterion for impurity: `information_gain`, `majority_error`, `gini_index`. (Default: All criteria are use)d. |

### Examples

1. **Running with default dataset (`car`) and directory (`./Data`):**
   ```bash
   ./run.sh
   ```

2. **Using `bank` and data directory:**
   ```bash
   ./run.sh --data bank --dir /path/to/Data
   ```

3. **Enabling "unknown" value handling**
   ```bash
   ./run.sh --data bank -M
   ```

4. **Setting max depth and impurity criterion and tree visualization:**
   ```bash
   ./run.sh --depth 5 --criterion information_gain -T
   ```

5. **Display help message:**
   ```bash
   ./run.sh --help
   ```