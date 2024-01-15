#!/bin/bash

ulimit -l

pwd

ls /edev

third_argument="$3"
extracted_value="${third_argument##*,}"

echo $extracted_value
sleep $extracted_value
echo "finished"