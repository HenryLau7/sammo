#!/bin/bash

# Define the variables
current_time=$(date +"%Y%m%d%H%M%S")
task="MATH"
method="apo"
# method="sammo"
marker="Mistral"

# Execute the python command with the defined variables
nohup python instruction_tuning_sammo.py \
    --task ${task} \
    --method ${method} \
    --llm vllm \
    --marker ${marker} \
    > log/${task}/${method}-${marker}-${current_time}.log 2>&1 &
