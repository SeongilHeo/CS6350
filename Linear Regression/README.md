# Homework 2 - Linear Regression
CS 5350/6350: Machine Learning Fall 2024  

## Setup

1. Clone the repository.
   ```bash
   git clone https://github.com/SeongilHeo/CS6350.git
   cd Linear Regression
   ```

2. Place Dataset only `train.csv` and `test.csv` according to the file structure below.  
   ```
   .
   ├── Data
   │   ├── bank
   │   ├── bool
   │   ├── concrete
   │   │   ├── data-desc.txt
   │   │   ├── test.csv
   │   │   └── train.csv
   │   └── tennis
   ├── Ensemble Learning
   │   ├── README.md
   │   ├── gradient.py
   │   ├── run.py
   │   └── run.sh
   │   ...
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
| `--dir DIR`            | Directory of the data folder. (Default:`./Data`. Example: `/path/to/Data`). |
| `--data DATA`          | Choose Dataset: `car`, `bank`. (Default: `car`).                             |
| `-M`, `--model`        | hoose batch or stochastic Gradient Descent. (options: batch, stochastic)      |
| `-R`, `--r`            | Set learning rate. (Optional, example: 0.01)                                 |
| `-q`                   | Choose question number. (options: 4a, 4b, 4c)                                 |


### Examples

1. **Running with default dataset (`concreate`) and directory (`../Data`) and `batch` stochastic model:**
   ```bash
   ./run.sh
   ```

2. **Using `other` dataset and data directory:**
   ```bash
   ./run.sh --data other --dir /path/to/Data
   ```
3. **Choosing gradient descent model (`stochastic`, `batch`):**
   ```bash
   ./run.sh --model stochastic
   ```

4. **Setting learning rate of model:**
   ```bash
   ./run.sh --model stochastic --r 0.01
   ```

5. **Running with # Question:**
   ```bash
   ./run.sh -q 4a
   ```

6. **Display help message:**
   ```bash
   ./run.sh --help
   ```

### Notes

You can run this program according to the qeustion  for grading. 

1. Place the dataset in the correct location.
2. Run `run.sh` with -q argument based on the problem. (Refer to example 5.)
   There are three options: 4a, 4b, and 4c.
