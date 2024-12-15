#!/bin/bash

chmod +x run.py

# Define default values for the arguments
DIR="../Data"
DATA="banknote"
Q=""

display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help         Display this help message and exit"
    echo "  --dir DIR          Directory of Data folder. (Default: ../Data, example: /path/to/Data)"
    echo "  --data DATA        Choose dataset: banknote. (Default: banknote)"
    echo "  -q Q               Choose question number. (options: 2a, 2b, 2c, 2e)"
    echo
    echo "Example:"
    echo " ./run.sh --dir /path/to/Data -q 2a"
    exit 0
}

# Parse the options passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) display_help ;;
        --dir) DIR="$2"; shift ;;
        --data) DATA="$2"; shift ;;
        -q) Q="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; display_help ;;
    esac
    shift
done

# Build the command
CMD="python3 run.py --dir $DIR --data $DATA"

if [ -n "$Q" ]; then
    CMD="$CMD -q $Q"
fi

echo "Executing: $CMD"
echo "------------------------------ [Start] -------------------------------"
$CMD
