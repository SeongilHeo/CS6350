#!/bin/bash

chmod +x run.py

# Define default values for the arguments
DATA="car"
DIR="./Data"
MISS=false
TREE=false
DEPTH=""
CRITERION=""

display_help() {
    echo "Usage: run.sh [options]"
    echo
    echo "Options:"
    echo "  --help               Show this help message and exit"
    echo "  --data DATA          Choose Dataset: car, bank. (Default: car)"
    echo "  --dir DIR            Directory of Data folder. (Default: ./Data)"
    echo "  -M, --miss           Enable this flag to handle 'unknown' values"
    echo "  -T, --tree           Enable this flag to visualize the tree"
    echo "  -D DEPTH, --depth DEPTH"
    echo "                       Set ID3 max depth. (Default: 1 to Max)"
    echo "  -C CRITERION, --criterion CRITERION"
    echo "                       Set criterion for impurity: information_gain, majority_error, gini_index."
    echo
    echo "Example:"
    echo "  ./run.sh --data bank --dir /path/to/Data -M -T --depth 5 --criterion information_gain"
    exit 0
}

# Parse the options passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help) display_help ;;  # Call the help function and exit
        --data) DATA="$2"; shift ;;
        --dir) DIR="$2"; shift ;;
        -M|--miss) MISS=true ;;
        -T|--tree) TREE=true ;;
        -D|--depth) DEPTH="$2"; shift ;;
        -C|--criterion) CRITERION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Build the command
CMD="python run.py --data $DATA --dir $DIR"

# Add optional flags and parameters
if [ "$MISS" = true ]; then
    CMD="$CMD -M"
fi

if [ "$TREE" = true ]; then
    CMD="$CMD -T"
fi

if [ -n "$DEPTH" ]; then
    CMD="$CMD --depth $DEPTH"
fi

if [ -n "$CRITERION" ]; then
    CMD="$CMD --criterion $CRITERION"
fi

# Print and execute the command
echo "Executing: $CMD"# Parse the options passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --help) display_help ;;  # Call the help function and exit
        --data) DATA="$2"; shift ;;
        --dir) DIR="$2"; shift ;;
        -M|--miss) MISS=true ;;
        -T|--tree) TREE=true ;;
        -D|--depth) DEPTH="$2"; shift ;;
        -C|--criterion) CRITERION="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Build the command
CMD="python run.py --data $DATA --dir $DIR"

# Add optional flags and parameters
if [ "$MISS" = true ]; then
    CMD="$CMD -M"
fi

if [ "$TREE" = true ]; then
    CMD="$CMD -T"
fi

if [ -n "$DEPTH" ]; then
    CMD="$CMD --depth $DEPTH"
fi

if [ -n "$CRITERION" ]; then
    CMD="$CMD --criterion $CRITERION"
fi

# Print and execute the command
echo "Executing: $CMD"
echo "------------------------------ [Start] -------------------------------"
$CMD
