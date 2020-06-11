#!/bin/bash
gpus=1

python main.py --gpu_id ${gpus} 2>&1 | tee log.txt
