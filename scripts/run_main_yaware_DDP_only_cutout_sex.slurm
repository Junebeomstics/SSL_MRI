#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 6:00:00
#SBATCH --nodes 32
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J sex_step
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

srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_only_cutout_steplr_sex_lr1e-4 --tb_dir ./tb_yaware_only_cutout_steplr_sex_lr1e4 --batch_size 64 --tf cutout --label_name sex --lr_policy step --lr_decay_iters 20 --gamma 0.5 --lr 1e-4 --kernel XOR

