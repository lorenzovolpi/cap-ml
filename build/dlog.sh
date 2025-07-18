#!/bin/bash

HOST="bcuda"
LOGFILE="quacc"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift
            shift
            ;;
        --local)
            HOST=""
            shift
            ;;
        -m|--module)
            LOGFILE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $HOST == "bcuda" || $HOST == "nity" ]]; then
    WORKDIR="quacc"
elif [[ $HOST == "dgx" ]]; then
    WORKDIR="raid/quacc"
else
    WORKDIR="."
fi


if [[ $1 == "-l" ]]; then
    $HOST cat "$WORKDIR/output/${LOGFILE}.log" | bat -l syslog
else
    $HOST tail -f -n +0 "$WORKDIR/output/${LOGFILE}.log" | bat -P -l syslog
fi
