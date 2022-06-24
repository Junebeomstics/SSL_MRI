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
#SBATCH -J gps_rescaled
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

# srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_only_cutout_step_sigma_intel_gps --tb_dir ./tb_yaware_only_cutout_step_sigma_intel_gps --batch_size 64 --tf cutout --label_name intelligence_gps --lr_policy step --lr_decay_iters 20 --lr 1e-3 --sigma 0.000005 --gamma 0.5

srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_only_cutout_step_intel_gps_rescaled --tb_dir ./tb_yaware_only_cutout_step_intel_gps_rescaled --batch_size 64 --tf cutout --label_name intelligence_gps --lr_policy step --sigma 5 --gamma 0.5
