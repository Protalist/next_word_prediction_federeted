#!/bin/bash

if [ "$1" = "" ]; then
    X=15
else
    X=$1
fi

command=""
for Y in $(seq $X)
do 
    name="clint-$Y"

    command="${command}; nt --title ${name} python3   new_client_random.py ${X} ${Y} "
done
if [ "$2" = "" ]; then
    cmd.exe /c wt.exe -w 0 nt --title server python3 new_server.py $X $command
    #cmd.exe /c wt.exe -w 0 nt $X $command
else
    cmd.exe /c wt.exe -w 0 $command
fi
#