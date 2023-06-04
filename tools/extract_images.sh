#!/bin/bash

outputDirectory=$(printf 'raw_data')
mkdir -pv "$outputDirectory"

for FILE in ./*.mp4
do
  if [ -f "$FILE" ]; then

    echo "Generate Images from $FILE"
    IFS='_'
    read -a strarr <<< "$FILE"
    time="_${strarr[4]}_${strarr[5]}_${strarr[6]}"
    IFS='.'
    read -a strarr <<< "$time"
    time="${strarr[0]}"
    fileName=$(printf "${outputDirectory}/frame%%06d${time}.jpg")
    start=$(ls "$outputDirectory" | wc -l)
    #ffmpeg -i "$FILE" -q:v 1 -hide_banner -loglevel error -f image2 -start_number "$start" "$fileName"
    ffmpeg -i "$FILE" -q:v 1 -vf "select=not(mod(n\,20))" -vsync vfr -q:v 2 -hide_banner -loglevel error -f image2 -start_number "$start" "$fileName"

  fi
done
