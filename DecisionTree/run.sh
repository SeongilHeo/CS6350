#!/bin/bash

# Commnet: 
# $1: dataset (필수)
# $2: handle unknonw  (optinal)

chmod +x run.py



if [ -z "$1" ]
then
    echo "run car datraset"
    arg1="car"
else
    arg1="$1"
fi

if [ -z "$2" ]
then
    arg2="N"
else
    arg2="$2"
fi

if [ -z "$3" ]
then
    echo "Data dir defualt=./Data/"
    arg3="./Data/"
else
    arg3="$3"
fi

python3 run.py "$arg1" "$arg2" "$arg3"