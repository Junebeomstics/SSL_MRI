#!/bin/bash

#SBATCH --job-name=MCIs500
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=titan
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./shell/220513_MCI_strat_500.txt

source /home/wonyoungjang/.bashrc

python3 main.py --mode finetuning --task_name MCI --task_target_num 500 --stratify strat
