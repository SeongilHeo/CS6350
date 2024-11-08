This is a machine learning library developed by Seongil Heo for CS5350/6350 in University of Utah.

## Homework 1 - Decision Tree
### Options

| Option                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--help`               | Show this help message and exit.                                             |
| `--data DATA`          | Choose Dataset: `car`, `bank`. (Default: `car`).                             |
| `--dir DIR`            | Directory of the data folder. (Default:`./Data`. Example: `/path/to/Data`). |
| `-M`, `--miss`         | Enable this flag to handle "unknown" values in the data.                     |
| `-T`, `--tree`         | Enable this flag to visualize the decision tree.                             |
| `-D DEPTH`, `--depth DEPTH` | Set the max depth of the ID3 decision tree. (Default: 1 to Max).           |
| `-C CRITERION`, `--criterion CRITERION` | Set the criterion for impurity: `information_gain`, `majority_error`, `gini_index`. (Default: All criteria are used). |
## Homework 2 - Ensemble Learning
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

## Homework 2 - Linear Regression
### Options

| Option                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `--help`               | Show this help message and exit.                                             |
| `--dir DIR`            | Directory of the data folder. (Default:`./Data`. Example: `/path/to/Data`). |
| `--data DATA`          | Choose Dataset: `car`, `bank`. (Default: `car`).                             |
| `-M`, `--model`        | hoose batch or stochastic Gradient Descent. (options: batch, stochastic)      |
| `-R`, `--r`            | Set learning rate. (Optional, example: 0.01)                                 |
| `-q`                   | Choose question number. (options: 4a, 4b, 4c)                                 |
## Homework 3 - Perceptron
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
