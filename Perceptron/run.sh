#!/bin/bash

chmod +x run.py

# Define default values for the arguments
DIR="../Data"
DATA="banknote"
MODEL="standard"
EPOCH="10"
LEARNING_RATE="0.01"
Q=""
display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help   Display this help message and exit"
    echo "  --dir        Directory of Data folder. (Default: ../Data, example: /path/to/Data)"
    echo "  --data       Choose Dataset: banknote. (Default: banknote)"
    echo "  -M, --model  Choose model. (options: standard, voted, average)"
    echo "  -T, --t      Set maximum epoch. (Default: 10)"
    echo "  -R, --r      Set learning rate. (Default: 0.01)"
    echo "  -q,          Choose question number. (options: 4a, 4b, 4c)"
    echo "Example:"
    echo " ./run.sh --dir /path/to/Data --data banknote -M voted -R 0.01 -T 20"
    exit 0
}

# Parse the options passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help) display_help ;;  
        -q) Q="$2"; shift ;;
        --dir) DIR="$2"; shift ;;
        --data) DATA="$2"; shift ;;
        -M|--model) MODEL="$2"; shift ;;
        -T|--t) EPOCH="$2"; shift ;;
        -R|--r) LEARNING_RATE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; display_help ;;
    esac
    shift
done

# Build the command
CMD="python3 run.py --dir $DIR --data $DATA"

# Add optional flags and parameters
if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi

if [ -n "$EPOCH" ]; then
    CMD="$CMD --t $EPOCH"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD --r $LEARNING_RATE"
fi

if [ -n "$Q" ]; then
    CMD="$CMD -q $Q"
fi

echo "Executing: $CMD"
echo "------------------------------ [Start] -------------------------------"
$CMD