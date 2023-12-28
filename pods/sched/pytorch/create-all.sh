#!/bin/bash

# Loop through each YAML file in the folder
for file in ./*.yaml; do
    if [ -f "$file" ]; then
        kubectl create -f $file
    fi
done
