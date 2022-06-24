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
#SBATCH -J simclr_cutout
#SBATCH --chdir=../
#SBATCH -o logs/%j-%x.out
set +x

# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

module load python
conda activate 3DCNN

env | grep SLURM

srun python main.py --mode pretraining --framework simclr --ckpt_dir ./checkpoint_simclr_only_cutout_220607 --tb_dir ./tb_simclr_only_cutout_220607 --batch_size 64 --tf cutout --label_name age

