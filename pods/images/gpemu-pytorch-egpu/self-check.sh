#!/bin/bash

ulimit -l

pwd

ls /edev

print_interval=60
while true
do
  echo "This message is printed every $print_interval second."
  sleep $print_interval  # Sleep for 5 second
done