#!/bin/bash

chmod +x run.py

# Define default values for the arguments
DIR="../Data"
DATA="banknote"
MODEL="primal"
KERNEL=""
SCHEDULE="A"
LEARNING_RATE="0.1"
A="1"
C=$(echo "scale=6; 100/873" | bc)  # Calculate default value for C
EPOCH="100"
GAMMA=""
Q=""

display_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help         Display this help message and exit"
    echo "  --dir DIR          Directory of Data folder. (Default: ../Data, example: /path/to/Data)"
    echo "  --data DATA        Choose dataset: banknote. (Default: banknote)"
    echo "  -M, --model MODEL  Choose SVM model. (options: primal, dual. Default: primal)"
    echo "  -K, --kernel       Choose kernel mode. (options: gaussian. Default: None)"
    echo "  -S, --schedule     Choose schedule type. (options: A, B. Default: None)"
    echo "  -R, --r            Set learning rate. (Default: 0.1)"
    echo "  -A, --a            Set hyperparameter of schedule. (Default: 1)"
    echo "  -C, --c            Set hyperparameter of model. (Default: 100/873)"
    echo "  -E, --epoch        Set number of epochs. (Default: 100)"
    echo "  -G, --gamma        Set gamma for Gaussian kernel. (Default: 1)"
    echo "  -q Q               Choose question number. (options: 2a, 2b, 3a, 3b, 3c)"
    echo
    echo "Example:"
    echo " ./run.sh --dir /path/to/Data --data banknote -M dual -K gaussian -C 1"
    exit 0
}

# Parse the options passed to the script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) display_help ;;
        --dir) DIR="$2"; shift ;;
        --data) DATA="$2"; shift ;;
        -M|--model) MODEL="$2"; shift ;;
        -K|--kernel) KERNEL="$2"; shift ;;
        -S|--schedule) SCHEDULE="$2"; shift ;;
        -R|--r) LEARNING_RATE="$2"; shift ;;
        -A|--a) A="$2"; shift ;;
        -C|--c) C="$2"; shift ;;
        -E|--epoch) EPOCH="$2"; shift ;;
        -G|--gamma) GAMMA="$2"; shift ;;
        -q) Q="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; display_help ;;
    esac
    shift
done

# Build the command
CMD="python3 run.py --dir $DIR --data $DATA -M $MODEL"

# Add optional flags and parameters
if [ -n "$KERNEL" ]; then
    CMD="$CMD -K $KERNEL"
fi

if [ -n "$SCHEDULE" ]; then
    CMD="$CMD -S $SCHEDULE"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD -R $LEARNING_RATE"
fi

if [ -n "$A" ]; then
    CMD="$CMD -A $A"
fi

if [ -n "$C" ]; then
    CMD="$CMD -C $C"
fi

if [ -n "$EPOCH" ]; then
    CMD="$CMD -E $EPOCH"
fi

if [ -n "$GAMMA" ]; then
    CMD="$CMD -G $GAMMA"
fi

if [ -n "$Q" ]; then
    CMD="$CMD -q $Q"
fi

echo "Executing: $CMD"
echo "------------------------------ [Start] -------------------------------"
$CMD
