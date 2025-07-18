#!/bin/bash

SERVER="bcuda"
MODULE="results"
TYPE="output"
daemon=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            SERVER="$2"
            shift
            shift
            ;;
        -d|--daemon)
            daemon=1
            shift
            ;;
        -m|--module)
            MODULE="$2"
            shift
            shift
            ;;
        -t|--type)
            TYPE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

if [[ $SERVER == "dgx" ]]; then
    HOST="lorenzovolpi@edge-nd1.isti.cnr.it"
    WORKDIR="raid/quacc"
elif [[ $SERVER == "bcuda" || $SERVER == "barracuda" ]]; then
    HOST="volpi@barracuda.isti.cnr.it"
    WORKDIR="quacc"
elif [[ $SERVER == "nity" || $SERVER == "trinity" ]]; then
    HOST="lorenzo.volpi@trinity.isti.cnr.it"
    WORKDIR="quacc"
fi

FROM_WORKDIR="$WORKDIR/$TYPE"
LOCAL_WORKDIR="/home/lorev/quacc/$TYPE"

sync_res() {
    rsync --info=progress2 -a "$HOST:$FROM_WORKDIR/$MODULE" $LOCAL_WORKDIR
}

daemon_sync_res() {
    while true; do
        sync_res
        sleep 10
    done
}

if [[ daemon -gt 0 ]]; then
    daemon_sync_res & disown
else
    sync_res
fi
