import numpy as np
from dataset import MRIDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from simclr import SimCLR
from torch.nn import CrossEntropyLoss
from losses import GeneralizedSupervisedNTXenLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
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

    model = SimCLR(net, loader_train, loss, config)

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        model.fine_tuning()




