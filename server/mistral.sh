#!/bin/bash

# Define the command to run
CMD="vllm serve /home/aiscuser/Mistral-7B-v0.1 --dtype auto --api-key token"

# Infinite loop to keep restarting the command if it crashes
while true
do
    echo "Starting vllm server..."
    
    # Run the command
    $CMD
    
    # # Check the exit status of the command
    # if [ $? -eq 0 ]; then
    #     echo "vllm server exited normally. Exiting the script."
    #     break
    # else
    #     echo "vllm server crashed. Restarting..."
    # fi
    
    # Optional: wait for a few seconds before restarting to avoid rapid restarts
    sleep 5
done
