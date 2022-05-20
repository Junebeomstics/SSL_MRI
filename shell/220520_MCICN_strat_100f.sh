#!/bin/bash

#SBATCH --job-name=MCICNs1f
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=3090
#SBATCH --time=0-12:00:00
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./shell/220520_MCICN_strat_100f.txt

source /home/wonyoungjang/.bashrc

python3 main.py --mode finetuning --task_name MCICN --task_target_num 100 --stratify strat
