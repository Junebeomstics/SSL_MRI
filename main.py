import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from dataset import MRIDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING
# ADNI
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--task_name", type=str, required=False, # ADNI
                        help="Set the name of the fine-tuning task.")
    parser.add_argument("--task_target_num", type=int, required=False, # ADNI
                        help="Set the number of training samples for the fine-tuning task.")
    parser.add_argument("--stratify", type=str, choices=["strat", "balan"], required=False, # ADNI
                        help="Set training samples stratified or not.")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)
    
    ### ADNI
    if config.mode == PRETRAINING:
        task_name = 'no' # no fine-tuning
        task_target_num = 0 # no fine-tuning
        stratify = "strat" # no fine-tuning
        dataset_train = MRIDataset(config, task_name, task_target_num, stratify, training=True)
        dataset_val = MRIDataset(config, task_name, task_target_num, stratify, validation=True)
        dataset_test = MRIDataset(config, task_name, task_target_num, stratify, test=True)
    elif config.mode == FINE_TUNING:
        dataset_train = MRIDataset(config, args.task_name, args.task_target_num, args.stratify, training=True)
        dataset_val = MRIDataset(config, args.task_name, args.task_target_num, args.stratify, validation=True)
        dataset_test = MRIDataset(config, args.task_name, args.task_target_num, args.stratify, test=True)
    ###

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

    ### ADNI
    loader_test = DataLoader(dataset_test,
                             batch_size=1,
                             sampler=RandomSampler(dataset_test),
                             collate_fn=dataset_test.collate_fn,
                             pin_memory=config.pin_mem,
                             num_workers=config.num_cpu_workers
                             )
    ###

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

    if config.mode == PRETRAINING:
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, task_name, task_target_num, stratify) # ADNI
    else:
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_name, args.task_target_num, args.stratify) # ADNI

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        outGT, outPRED = model.fine_tuning() # ADNI
        #print('outGT:', outGT)
        #print('outPRED:', outPRED)
    
    ### ADNI
    if config.mode == FINE_TUNING:
        outAUROC = []
        outGTnp = outGT.cpu().numpy()
        outPREDnp = outPRED.cpu().numpy()
        outAUROC = roc_auc_score(outGTnp[:, 1], outPREDnp[:, 1]) # idx 1 is case label
        print('<<< {0} Test Results: AUROC >>>'.format(args.task_name))
        print('{:.4f}\n'.format(outAUROC))
        
        fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, 1], outPRED.cpu()[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
        plt.title('ROC for {0}'.format(args.task_name))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig('./figs/ADNI_{0}_{1}_{2}_ROC.png'.format(args.task_name, args.stratify, args.task_target_num), dpi = 100)
        plt.close()
    ###
