#!/bin/bash

#SBATCH --job-name=ADMCIs5
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./shell/220520_ADMCI_strat_500.txt

source /home/wonyoungjang/.bashrc

python3 main.py --mode finetuning --task_name ADMCI --task_target_num 500 --stratify strat