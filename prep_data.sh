#!/bin/bash
input="./meta.txt"
img_dir="./MeGlass_120x120"
while IFS= read -r line
do
    filename=$(echo "$line" | cut -d" " -f 1)
    class=$(echo "$line" | cut -d" " -f 2)
    if [ $class -eq 0 ]
    then
        mv -n $img_dir/$filename $img_dir/no_glasses/
    else
        mv -n $img_dir/$filename $img_dir/yes_glasses/
    fi
done < "$input"
