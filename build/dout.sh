#!/bin/bash

HOST="bcuda"
OUTPUT_DIR="output"
OUTPUT_FILE="quacc"
FILTER=false
TAIL=false


while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST=$2
            shift
            shift
            ;;
        -t|--tail)
            TAIL=true
            shift 
            ;;
        -m|--module)
            OUTPUT_FILE=$2
            shift
            shift
            ;;
        -d|--dir)
            OUTPUT_DIR=$2
            shift
            shift
            ;;
        -f|--filter)
            FILTER=true
            shift
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $HOST == "bcuda" || $HOST == "nity" ]]; then
    PYTHON_PATH="miniconda3/bin/python"
    WORKDIR="quacc"
elif [[ $HOST == "dgx" ]]; then
    PYTHON_PATH="raid/miniconda3/bin/python"
    WORKDIR="raid/quacc"
fi


if [[ $TAIL == true ]]; then
    $HOST tail -n +0 -f "$WORKDIR/$OUTPUT_DIR/${OUTPUT_FILE}.out" | bat -P
else
    if [[ $FILTER == true ]]; then
        $HOST "$PYTHON_PATH" "$WORKDIR/filter_out.py" "$WORKDIR/$OUTPUT_DIR/${OUTPUT_FILE}.out" | bat
    else
        $HOST cat "$WORKDIR/$OUTPUT_DIR/${OUTPUT_FILE}.out" | bat
    fi
fi
