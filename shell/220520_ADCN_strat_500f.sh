#!/bin/bash

#SBATCH --job-name=ADCNs5f
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=3090
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./shell/220520_ADCN_strat_500f.txt

source /home/wonyoungjang/.bashrc

python3 main.py --mode finetuning --task_name ADCN --task_target_num 500 --stratify strat
