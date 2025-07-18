#!/bin/bash

if [[ $(hostname) == "barracuda.isti.cnr.it" ]]; then
    ENVFILE="bcuda.env"
elif [[ $(hostname) == "dgx-a100" ]]; then
    ENVFILE="dgx.env"
else
    ENVFILE="local.env"
fi

export $(cat "${ENVFILE}" | xargs)

OUTPUT_FILE="quacc.out"
MODULE="quacc.experiments.run"
FRONT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpus)
            export QUACC_N_JOBS=$2
            shift
            shift 
            ;;
        -m|--module)
            MODULE=$2
            shift
            shift
            ;;
        -o|--out)
            OUTPUT_FILE=$2
            shift
            shift
            ;;
        -env)
            shift
            while [[ $1 = *\=* ]]; do
                name=${1%%=*}
                val_start=$(("${#name}"+1))
                tot_len="${#1}"
                val=${1:$val_start:$tot_len}
                export $(echo "$name=$val" | xargs)
                shift
            done
            ;;
        -f|--front)
            FRONT=true
            shift
            ;;
        --stop)
            pkill -f joblib -u $USER
            sleep 2
            pkill -f $MODULE -u $USER
            exit 0
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $FRONT == true ]]; then
    poetry run python -um $MODULE
else
    poetry run python -um $MODULE &> "$QUACC_OUT_DIR/$OUTPUT_FILE" & disown
fi
    
