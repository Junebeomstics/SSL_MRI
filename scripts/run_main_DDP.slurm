#!/bin/bash
#SBATCH -A m3898_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH -J DDP_wd_barrier
set +x


# -C : constraints 
#-n : ntasks
#-c : --cpus-per-task
#-G : --gpus-per-task

module load python
conda activate 3DCNN

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
# export MASTER_PORT=12340
# export WORLD_SIZE=8 # total number of processes(gpus)

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
#echo "NODELIST="${SLURM_NODELIST}
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#echo "MASTER_ADDR="$MASTER_ADDR

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py  --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_DDP --batch_size 64
srun python main.py --mode pretraining --framework yaware --ckpt_dir ./checkpoint_yaware_DDP --DDP on --batch_size 16
