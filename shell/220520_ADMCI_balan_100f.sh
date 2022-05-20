#!/bin/bash

#SBATCH --job-name=ADMCIb1f
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=3090
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./shell/220520_ADMCI_balan_100f.txt

source /home/wonyoungjang/.bashrc

python3 main.py --mode finetuning --task_name ADMCI --task_target_num 100 --stratify balan
