#!/bin/bash

#SBATCH --job-name=ADCN
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=3090
#SBATCH --time=0-12:00:00
#SBATCH --mem=20000MB
#SBATCH --cpus-per-task=8
#SBATCH -o ./shell/ADCN_BHBya64c_CV.txt

source /home/wonyoungjang/.bashrc

echo "Used BHBya64c.pth (DenseNet121_BHB-10K_yAwareContrastive.pth) for pretrained weights. (meta: age)"
echo "Used self.model='DenseNet', self.nb_epochs=100, self.tf='cutout', self.batch_size = 8"
echo "self.input_size=(1,80,80,80), self.lr=1e-4, self.weight_decay=5e-5, self.patience=20"
echo ""
echo "--train_num 100 & --layer_control tune_all"
python3 main_cv.py --train_num 100 --task_name AD/CN --layer_control tune_all --random_seed 0
echo "--train_num 100 & --layer_control freeze"
python3 main_cv.py --train_num 100 --task_name AD/CN --layer_control freeze --random_seed 0
echo "--train_num 300 & --layer_control tune_all"
python3 main_cv.py --train_num 300 --task_name AD/CN --layer_control tune_all --random_seed 0
echo "--train_num 300 & --layer_control freeze"
python3 main_cv.py --train_num 300 --task_name AD/CN --layer_control freeze --random_seed 0
