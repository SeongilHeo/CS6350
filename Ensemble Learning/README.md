# Homework 2 - Ensemble Model
CS 5350/6350: Machine Learning Fall 2024  

## Setup

1. Clone the repository.
   ```bash
   git clone https://github.com/SeongilHeo/CS6350.git
   cd Ensemble Learning
   ```

2. Place Dataset only `train.csv` and `test.csv` according to the file structure below.  
   ```
   .
   ├── Data
   │   ├── bank
   │   │   ├── data-desc.txt
   │   │   ├── test.csv
   │   │   └── train.csv
   │   ├── bool
   │   ├── car
   │   ├── concrete
   │   └── tennis
   ├── Ensemble Learning
   │   ├── Ensemble.py
   │   ├── ID3.py
   │   ├── README.md
   │   ├── run.py
   │   ├── run.sh
   │   ├── utils.py
   │   └── visualize.py
   │  ...
   └── README.md  
    ```

3. Make the `run.sh` script executable.
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
| `--dir DIR`            | Directory of the data folder. (Default:`../Data`. Example: `/path/to/Data`). |
| `--data DATA`          | Choose Dataset: `bank`. (Default: `bank`).                             |
| `-M`, `--model`        | Choose ensemble model: `AdaBoost`, `BaggedTrees`, `RandomForest`. (Default: `AdaBoost`)        |
| `-N NUM`, `--num NUM`  | Set number or range of # trees . (Default: 50, ex: 50, 1-50) |
| `--ratio RATIO`        | Set ratio of boostrapping. (Default: 1)                        |
| `--seed SEED  `        | Set random seed for boostrapping. (Default: None)                       |
| `--nattr NATTR  `        | Set bootstraping number of attribute for Randomforest. (Default: 2)                    |
| `-q Q  `               | hoose question number. (options: 2a, 2b, 2c, 2d, 2e)                   |


### Examples

1. **Running with default dataset (`car`) and directory (`./Data`):**
   ```bash
   ./run.sh
   ```

2. **Using `bank` and data directory:**
   ```bash
   ./run.sh --data bank --dir /path/to/Data
   ```

3. **Choosing Ensemble learning model:**
   ```bash
   ./run.sh --model Adaboost
   ```

4. **Setting number of trees (or stumps):**
   ```bash
   ./run.sh --model Adaboost --num 1-50
   ./run.sh --model Adaboost --num 50
   ```

5. **Setting boostraping ratio for bagged trees:**
   ```bash
   ./run.sh --model BaggedTrees --ratio 0.3
   ```
6. **Setting random seed for boostratping  for bagged trees:**
   ```bash
   ./run.sh --model BaggedTrees --ratio 0.3 --seed 42
   ```

7. **Setting number of attributes for random forest:**
   ```bash
   ./run.sh --model RandomForest --nattr 2
   ```

8. **Running with # Question:**
   ```bash
   ./run.sh -q 2a
   ```

9. **Display help message:**
   ```bash
   ./run.sh --help
   ```

### Notes

You can run this program according to the qeustion  for grading. 

1. Place the dataset in the correct location.
2. Run `run.sh` with -q argument based on the problem. (Refer to example 8.)
   There are three options: 2a, 2b, 2c, 2d and 4e.
