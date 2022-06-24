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
#SBATCH -J b64_simclr
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

srun python main.py --mode pretraining --framework simclr --tf all_tf --ckpt_dir ./checkpoint_simclr_b64_220612  --tb_dir ./tb_simclr_b64_220612 --batch_size 64

