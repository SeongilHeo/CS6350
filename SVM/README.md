# Homework 4 - SVM
CS 5350/6350: Machine Learning Fall 2024  

## Setup

1. Clone the repository.
   ```bash
   git clone https://github.com/SeongilHeo/CS6350.git
   cd SVM
   ```

2. Place Dataset only `train.csv` and `test.csv` according to the file structure below.  
   ```
   .
   ├── Data
   │   ├── bank
   │   ├── bool
   │   ├── banknote
   │   │   ├── data-desc.txt
   │   │   ├── test.csv
   │   │   └── train.csv
   │   └── tennis
   ├── SVM
   │   ├── README.md
   │   ├── model.py
   │   ├── run.py
   │   ├── run.sh
   │   └── utils.py
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
| `--help`               | Show this help message and exit.|
| `--dir DIR`            | Directory of the data folder. (Default: `../Data`. Example: `/path/to/Data`).|
| `--data DATA`          | Choose dataset. (Default: `banknote`).|
| `-M`, `--model`        | Choose SVM model. (Options: `primal`, `dual`. Default: `primal`).|
| `-K`, `--kernel`       | Choose kernel mode. (Options: `gaussian`. Default: `None`).|
| `-S`, `--schedule`     | Choose learning rate schedule type. (Options: `A`, `B`. Default: `None`).|
| `-R`, `--learning_rate`| Set learning rate for training. (Default: `0.01`).|
| `-A`, `--a`            | Set hyperparameter for the schedule. (Default: `1`).|
| `-C`, `--c`            | Set the regularization hyperparameter for the model. (Default: `100/873`).|
| `-E`, `--epoch`        | Set the number of training epochs. (Default: `100`).|
| `-G`, `--gamma`        | Set the gamma parameter for the Gaussian kernel. (Default: `1`).|
| `-q`                   | Specify the question number for the task. (Options: `2a`, `2b`, `3a`, `3b`, `3c`. Default: `None`). |

---
### Examples

1. **Running with default dataset (`banknote`) and directory (`../Data`) and `banknote` stochastic model:**
   ```bash
   ./run.sh
   ```

2. **Using `other` dataset and data directory:**
   ```bash
   ./run.sh --data other --dir /path/to/Data
   ```
3. **Specifying the schduel for primal model:**
   ```bash
   ./run.sh --model primal --schuel A
   ```

4. **Specifying the gamma value for the Gaussian kernel**:
   ```bash
   ./run.sh --model dual -K gaussian -G 0.1
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
   There are five options: 2a, 2b, 3a, 3b, and 3c.


