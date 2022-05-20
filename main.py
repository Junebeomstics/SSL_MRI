import numpy as np
from dataset import MRIDataset, UKBDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss, NTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING
import pandas as pd
import os
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--framework", type=str, choices=["yaware", "simclr"], required=True,
                        help="select which framework to use !")
    
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='local rank for distributed training')
    
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode, args.framework)
    
    meta_data = pd.read_csv(config.label)
    subjects = os.listdir(config.data)
    
    if config.label_name == 'sex':
        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])]['sex'].values[0]) for subj in subjects]
    elif config.label_name == 'age':
        subj_meta = [(subj,meta_data[meta_data['eid']==int(subj[:7])]['age'].values[0]) for subj in subjects]
        
    #subj_meta is a list consisting of tuple (filename, label(int))
    
    random.shuffle(subj_meta)
    
    num_total = len(subj_meta)
    num_train = int(num_total*(1 - config.val_size))
    num_val = int(num_total*config.val_size)
    
    subj_train= subj_meta[:num_train]
    subj_val = subj_meta[num_train:]



    if config.mode == mode:
        dataset_train = UKBDataset(config, subj_train) #MRIDataset(config, training=True)
        dataset_val = UKBDataset(config,subj_val)  #MRIDataset(config, validation=True)
    else:
        ## Fill with your target dataset
        dataset_train = Dataset()
        dataset_val = Dataset()

    loader_train = DataLoader(dataset_train,
                              batch_size=config.batch_size,
                              sampler=RandomSampler(dataset_train),
                              collate_fn=dataset_train.collate_fn,
                              pin_memory=config.pin_mem,
                              num_workers=config.num_cpu_workers
                              )
    loader_val = DataLoader(dataset_val,
                            batch_size=config.batch_size,
                            sampler=RandomSampler(dataset_val),
                            collate_fn=dataset_val.collate_fn,
                            pin_memory=config.pin_mem,
                            num_workers=config.num_cpu_workers
                            )
    if config.mode == PRETRAINING:
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
    if config.mode == PRETRAINING:
        if config.framework == 'simclr':
            loss = NTXenLoss(temperature=config.temperature,return_logits=True)
        elif confing.framework == 'yaware':
            loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature,
                                              kernel='rbf',
                                              sigma=config.sigma,
                                              return_logits=True)

    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()

    model = yAwareCLModel(net, loss, loader_train, loader_val, config)

    if config.mode == PRETRAINING:
        if config.framework == 'simclr':
            model.pretraining_simclr()
        elif confing.framework == 'yaware':
            model.pretraining_yaware()
    else:
        model.fine_tuning()




