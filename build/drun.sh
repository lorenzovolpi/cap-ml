#!/bin/bash

# export env variables from docker.env
export $(cat docker.env | xargs)

IMAGE_NAME="lorenzovolpi/quacc_main"
CONTAINER_NAME="quacc_run"
CPUS=32
MODULE="exp.run"
OUTPUT_FILE="quacc.out"

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpus)
            CPUS=$2
            shift
            shift 
            ;;
        -m|--module)
            MODULE=$2
            shift
            shift
            ;;
        -n|--name)
            CONTAINER_NAME=$2
            shift
            shift
            ;;
        -o|--out)
            OUTPUT_FILE=$2
            shift
            shift
            ;;
        --stop)
            docker stop $CONTAINER_NAME
            echo "stopped"
            exit 0
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# RUN_CMD="python -m $MODULE"
RUN_CMD="python -um $MODULE &> $QUACC_OUT_DIR/$OUTPUT_FILE"

docker run -d \
-it \
--rm \
--user $(id -u):$(id -g) \
--name $CONTAINER_NAME \
--cpus $CPUS \
--env-file docker.env \
--mount type=bind,source="$(pwd)$QUACC_OUT_DIR",target=$QUACC_OUT_DIR \
--mount type=bind,source="$(pwd)$QUACC_SKLEARN_DATA",target=$QUACC_SKLEARN_DATA \
--mount type=bind,source="$(pwd)$QUACC_QUAPY_DATA",target=$QUACC_QUAPY_DATA \
--mount type=bind,source="$(pwd)$QUACC_DATA",target=$QUACC_DATA \
$IMAGE_NAME \
bash -c "$RUN_CMD"

