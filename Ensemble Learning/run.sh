#!/bin/bash

chmod +x run.py

# Define default values for the arguments
DIR="../Data"
DATA="bank"
MODEL="AdaBoost"
NUM="50"
RATIO=1
SEED=""
NUMATTR="2"

display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help   Display this help message and exit"
    echo "  --dir        Directory of Data folder. (Default: ../Data, example: /path/to/Data)"
    echo "  --data       Choose dataset: car, bank. (Default: bank)"
    echo "  --model      Choose ensemble model: AdaBoost, BaggedTrees. (Default: AdaBoost)"
    echo "  -N, --num    Set number or range of trees. (Default: 1-50, example: 50 or 1-50)"
    echo "  --ratio      Set ratio of bootstrapping. (Default: 1)"
    echo "  --seed       Set random seed for bootstrapping. (Default: None)"
    echo "  --nattr   Set bootstraping number of attribute for Randomforest. (Default: 2)"
    echo "  -q,          Choose question number. (options: 2a, 2b, 2c, 2d, 2e)"
    echo
    echo "Example:"
    echo " ./run.sh --dir /path/to/Data --data bank --model BaggedTrees -N 1-100 --ratio 0.8 --seed 42"
    exit 0
}

# Parse the options passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help) display_help ;;  # Call the help function and exit
        -q) Q="$2"; shift ;;
        --dir) DIR="$2"; shift ;;
        --data) DATA="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        -N|--num) NUM="$2"; shift ;;
        --ratio) RATIO="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --nattr) NUMATTR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Build the command
CMD="python run.py --dir $DIR --data $DATA --model $MODEL -N $NUM --ratio $RATIO"

# Add optional flags and parameters
if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

if [ -n "$NUMATTR" ]; then
    CMD="$CMD --nattr $NUMATTR"
fi

if [ -n "$Q" ]; then
    CMD="$CMD -q $Q"
fi


echo "Executing: $CMD"
echo "------------------------------ [Start] -------------------------------"
$CMD
