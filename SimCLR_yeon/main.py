import numpy as np
from dataset import MRIDataset
from torch import optim
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from simclr import SimCLR
from torch.nn import CrossEntropyLoss
from losses import GeneralizedSupervisedNTXenLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from torchvision import models
from models.resnet_simclr import ResNetSimCLR
from config import Config, PRETRAINING, FINE_TUNING
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
    parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
    
    args = parser.parse_args()

    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)

    if config.mode == mode: #pretraining
        loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature,
                                              kernel='rbf',
                                              sigma=config.sigma,
                                              return_logits=True)
        dataset_train = MRIDataset(config, training=True)
        dataset_val = MRIDataset(config, validation=True)
    else: #finetuning
        ## Fill with your target dataset
        dataset_train = Dataset()
        dataset_val = Dataset()
        loss = CrossEntropyLoss()

    #when we want to look at adni data
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
    #model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)    #main base resnet 18
    if config.model == "DenseNet":
        model = densenet121(mode="encoder", drop_rate=0.0)
    elif config.model == "UNet":
        model = UNet(config.num_classes, mode="simCLR")
    else:
        raise ValueError("Unkown model: %s"%config.model)
    model = densenet121(mode="encoder", drop_rate=0.0)
    optimizer = optim.Adam(model.parameters(), 0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader_train), eta_min=0,
                                                       last_epoch=-1)
    with torch.cuda.device(0):
        simclr = SimCLR(loss, model = model, config = config, loader_train = loader_train, loader_val = loader_val, optimizer=optimizer, scheduler=scheduler)
        simclr.train(loader_train)

    """
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
        loss = GeneralizedSupervisedNTXenLoss(temperature=config.temperature,
                                              kernel='rbf',
                                              sigma=config.sigma,
                                              return_logits=True)
        
    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()
        
    optimizer = optim.Adam(net.parameters(), 0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader_train), eta_min=0,
                                                           last_epoch=-1)
    model = SimCLR(loss, model = net, config = config, loader_train = loader_train, loader_val = loader_val, optimizer=optimizer, scheduler=scheduler)

    if config.mode == PRETRAINING:
        model.train(loader_train)
    else:
        model.fine_tuning()
    """




