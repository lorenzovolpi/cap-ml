#!/bin/bash

SERVER="bcuda"
while [[ $# -gt 0 ]]; do
  case $1 in
  -h | --host)
    SERVER="$2"
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

  DEST="$HOST:$WORKDIR"

  rsync -ai calibration quacc playground exp $DEST
  rsync -i README.md build/run.sh build/dgx.env build/filter_out.py pyproject.toml $DEST
elif [[ $SERVER == "bcuda" || $SERVER == "barracuda" ]]; then
  HOST="volpi@barracuda.isti.cnr.it"
  WORKDIR="quacc"

  DEST="$HOST:$WORKDIR"

  rsync -ai calibration quacc playground exp $DEST
  rsync -i README.md build/run.sh build/bcuda.env build/filter_out.py pyproject.toml $DEST
  rsync -ai qcdash $DEST
  rsync -i build/dash.sh build/dash.env $DEST
elif [[ $SERVER == "trinity"  || $SERVER == "nity" ]]; then
  HOST="lorenzo.volpi@trinity.isti.cnr.it"
  WORKDIR="quacc"

  DEST="$HOST:$WORKDIR"

  rsync -ai calibration quacc playground exp $DEST
  rsync -i README.md build/run.sh build/nity.env build/filter_out.py pyproject.toml $DEST
fi
