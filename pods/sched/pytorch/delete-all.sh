#!/bin/bash

# Loop through each YAML file in the folder
for file in user1/*.yaml; do
    if [ -f "$file" ]; then
        kubectl delete -f $file
    fi
done

for file in user2/*.yaml; do
    if [ -f "$file" ]; then
        kubectl delete -f $file
    fi
done

for file in user3/*.yaml; do
    if [ -f "$file" ]; then
        kubectl delete -f $file
    fi
done

for file in user4/*.yaml; do
    if [ -f "$file" ]; then
        kubectl delete -f $file
    fi
done