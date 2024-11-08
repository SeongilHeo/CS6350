# Homework 3 - Perceptron
CS 5350/6350: Machine Learning Fall 2024  

## Setup

1. Clone the repository.
   ```bash
   git clone https://github.com/SeongilHeo/CS6350.git
   cd Perceptron
   ```

2. Place Dataset only `train.csv` and `test.csv` according to the file structure below.  
   ```
   .
   ├── Data
   │   │   ...
   │   ├── banknote
   │   │   ├── data-desc
   │   │   ├── test.csv
   │   │   └── train.csv
   │   └── ...

   ├── DecisionTree
   │   ...
   ├── Perceptron
   │   ├── README.md
   │   ├── model.py
   │   ├── run.py
   │   ├── run.sh
   │   └── utils.py
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
| `--data DATA`          | Choose Dataset: `banknote`. (Default: `banknote`).                             |
| `-M`, `--model`        | Choose model. (Default: `standard`, Options: `standard`, `voted`, `average`)      |
| `-T EPOOCH`,`--t EPOOCH`| Set maximum epoch. (Default: 10)                                             |
| `-R Learningrate`,`--r Learningrate`| Set learning rate. (Default: 0.01)                                             |
| `-q`                   | Choose question number. (Options: 2a, 2b, 2c)                                 |


### Examples

1. **Running with default dataset (`bank-note`) and directory (`../Data`) and standard perceptron model:**
   ```bash
   ./run.sh
   ```

2. **Using `other` dataset and data directory:**
   ```bash
   ./run.sh --data other --dir /path/to/Data
   ```
3. **Choosing gradient descent model (`standard`,`voted`, `average`):**
   ```bash
   ./run.sh --model voted
   ```

4. **Setting maximum epooch of learning:**
   ```bash
   ./run.sh --model average --t 20
   ```

5. **Running with # Question:**
   ```bash
   ./run.sh -q 2a
   ```

6. **Display help message:**
   ```bash
   ./run.sh --help
   ```

### Notes

You can run this program according to the qeustion  for grading. 

1. Place the dataset in the correct location.
2. Run `run.sh` with -q argument based on the problem. (Refer to example 5.)
   There are three options: 2a, 2b, and 2c.
