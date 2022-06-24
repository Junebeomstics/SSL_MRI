import numpy as np
from dataset import MRIDataset, UKBDataset

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

#DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import DataParallel
import builtins


from utils import get_scheduler

from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config
import pandas as pd
import os
import random
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--framework", type=str, choices=["yaware", "simclr"], required=True)
    parser.add_argument("--kernel", type=str, default = 'rbf',choices=["rbf", "XOR"], help="rbf(continuous), XOR(binary categorical variable)")
    parser.add_argument("--ckpt_dir", type=str, default = './checkpoint',
                        help="select which dir to save the checkpoint!")
    parser.add_argument("--tb_dir", type=str, default = './tb',
                       help="select which dir to save the tensorboard log")
    parser.add_argument("--tf", type=str, default = 'all_tf',choices=['all_tf','cutout','crop'],
                       help="select which transforms to apply")
    parser.add_argument("--nb_epochs", type=int, default = 100, help="number of epochs")
    parser.add_argument("--lr", type=float, default = 1e-4, help="initial learning rate for optimizer")
    parser.add_argument("--label_name", type=str, default = 'age', choices= ['age', 'sex', 'intelligence_gps', 'intelligence'], help="target meta info")
    parser.add_argument("--lr_policy",type=str,default='None' ,choices=['lambda','step','multi-step','plateau','cosine','SGDR','None'], help='learning rate policy: lambda|step|multi-step|plateau|cosine|SGDR')
    parser.add_argument('--lr_decay_iters', type=int, default=10, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument("--gamma",default=0.1,type=float,help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument("--sigma",default=5,type=float,help='Hyperparameters for our y-Aware InfoNCE Loss depending on the meta-data at hand')                    
    
    
    # DDP configs:
    parser.add_argument('--world_size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='batch_size')
    
    args = parser.parse_args()
    

    config = Config(args)
    
    
    meta_data = pd.read_csv(config.label)
    
    
    
    subjects = sorted(os.listdir(config.data))
    
    if config.label_name == 'sex':
        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])]['sex'].values[0]) for subj in subjects]
    elif config.label_name == 'age':
        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])]['age'].values[0]) for subj in subjects]
    elif config.label_name == 'intelligence_gps':
        meta_data = meta_data.dropna(subset=['SCORE_auto'])

        #rescale to min & max of the age
        age_min = meta_data['age'].min()
        age_max = meta_data['age'].max()
        scaler = MinMaxScaler((age_min,age_max))
        meta_data['SCORE_auto'] = scaler.fit_transform(meta_data[['SCORE_auto']])
        print(f'the values of intelligence_gps are min-max scaled to {age_min} ~ {age_max}')

        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])]['SCORE_auto'].values[0]) for subj in subjects if int(subj[:7]) in meta_data['eid'].tolist()]

    elif config.label_name == 'intelligence':
        meta_data = meta_data.dropna(subset=['fluid'])

        #rescale to min & max of the age
        age_min = meta_data['age'].min()
        age_max = meta_data['age'].max()
        scaler = MinMaxScaler((age_min,age_max))
        meta_data['fluid'] = scaler.fit_transform(meta_data[['fluid']])
        print(f'the values of intelligence are min-max scaled to {age_min} ~ {age_max}')

        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])]['fluid'].values[0]) for subj in subjects if int(subj[:7]) in meta_data['eid'].tolist()] 
        
    
    print(f'training {len(subj_meta)} UKB subjects')
        
    #subj_meta is a list consisting of tuple (filename, label(int))
    
    random.shuffle(subj_meta)
    
    num_total = len(subj_meta)
    num_train = int(num_total*(1 - config.val_size))
    num_val = int(num_total*config.val_size)
    
    subj_train= subj_meta[:num_train]
    subj_val = subj_meta[num_train:]
    
    if config.mode == 'pretraining':
        if config.model == "DenseNet":
            net = densenet121(mode="encoder", drop_rate=0.0)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="simCLR")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    else:
        if config.model == "DenseNet":
            net = densenet121(mode="classifier", drop_rate=0.0, num_classes=config.num_classes)
        elif config.model == "UNet":
            net = UNet(config.num_classes, mode="classif")
        else:
            raise ValueError("Unkown model: %s"%config.model)
    
    
    
    def _get_sync_file():
        """Logic for naming sync file using slurm env variables"""
        sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
        os.makedirs(sync_file_dir, exist_ok=True)
        sync_file = 'file://%s/pytorch_sync.%s.%s' % (
            sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
        return sync_file
    
    ### DDP         
    # sbatch script에서 WORLD_SIZE를 지정해준 경우 (노드 당 gpu * 노드의 수)
    if "WORLD_SIZE" in os.environ:
        config.world_size = int(os.environ["WORLD_SIZE"])
    # 혹은 슬럼에서 자동으로 ntasks per node * nodes 로 구해줌
    elif 'SLURM_NTASKS' in os.environ:
        config.world_size = int(os.environ['SLURM_NTASKS'])
        
    config.distributed = config.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if config.distributed:
        if config.local_rank != -1: # for torch.distributed.launch
            config.rank = config.local_rank
            config.gpu = config.local_rank
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            config.rank = int(os.environ['SLURM_PROCID'])
            config.gpu = config.rank % torch.cuda.device_count()

        sync_file = _get_sync_file()
        dist.init_process_group(backend=config.dist_backend, init_method=sync_file,
                            world_size=config.world_size, rank=config.rank)
    else:
        config.rank = 0
        config.gpu = 0

    # suppress printing if not on master gpu
    if config.rank!=0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
        
    ### model 
    if config.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if config.gpu is not None:
            config.device = torch.device('cuda:{}'.format(config.gpu))
            torch.cuda.set_device(config.gpu)
            net.cuda(config.gpu)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[config.gpu], broadcast_buffers=False)
            net_without_ddp = net.module
        else:
            config.device = torch.device("cuda" if config.cuda else "cpu")
            net.cuda()
            net = torch.nn.parallel.DistributedDataParallel(net)
            model_without_ddp = net.module
    else:
        config.device = torch.device("cuda" if config.cuda else "cpu")
        net = DataParallel(net).to(config.device)        
            
    torch.backends.cudnn.benchmark = True        
    
    if config.mode == 'pretraining':
        dataset_train = UKBDataset(config, subj_train) #MRIDataset(config, training=True)
        dataset_val = UKBDataset(config,subj_val)  #MRIDataset(config, validation=True)
    else:
        ## Fill with your target dataset
        dataset_train = Dataset()
        dataset_val = Dataset()
        
    if config.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_val, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_val)
        

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=train_sampler,
                              collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=valid_sampler,
                            collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )
    if config.mode == 'pretraining':
        if config.framework == 'simclr':
            loss = NTXenLoss(temperature=config.temperature,return_logits=True)
        elif config.framework == 'yaware':
            loss = GeneralizedSupervisedNTXenLoss(config = config, temperature=config.temperature,
                                              kernel=config.kernel,
                                              sigma=config.sigma,
                                              return_logits=True)

    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()
    
    model = yAwareCLModel(net, loss, loader_train, loader_val, config)

    if config.mode == 'pretraining':
        if config.framework == 'simclr':
            model.pretraining_simclr()
        elif config.framework == 'yaware':
            model.pretraining_yaware()
    else:
        model.fine_tuning()




