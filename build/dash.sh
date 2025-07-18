#!/bin/bash
ENVFILE="build/dash.env"

if [[ $1 == "--kill" ]]; then
	pkill -f qcdash -u $USER &> /dev/null
	sleep 2
	echo "dash stopped"
	exit 0
fi

if [[ $(pgrep -f qcdash -u $USER) ]]; then
	echo "dash is already running"
else
	export $(cat "$ENVFILE" | xargs)
	poetry run gunicorn qcdash.app:server -b localhost:33421 &>"${HOME}/quacc/output/dash.out" & disown
	sleep 2
	echo "dash started"
fi


