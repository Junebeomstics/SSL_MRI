#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH --nodes 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J yclr
#SBATCH --chdir=../
#SBATCH -o logs/%j-%x.out
set +x

# UKBsa64c+UKBya64c

# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

module load python
conda activate 3DCNN

env | grep SLURM

srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_simclr_80_yaware_20 --tb_dir ./tb_simclr_only_cutout_220607 --batch_size 64 --tf cutout --label_name age --nb_epochs 120 --lr_policy step --lr 1e-3 --sigma 5

