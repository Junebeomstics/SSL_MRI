### ADNI
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import time
###
import numpy as np
from dataset import ADNI_Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from yAwareContrastiveLearning import yAwareCLModel
from losses import GeneralizedSupervisedNTXenLoss
from torch.nn import CrossEntropyLoss
from models.densenet import densenet121
from models.unet import UNet
import argparse
from config import Config, PRETRAINING, FINE_TUNING
### ADNI
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
###

if __name__ == "__main__":
    start_time = time.time() # ADNI
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--task_names", type=str, choices=["ADCN", "MCICN", "ADMCI"], required=False, # ADNI
                        help="Set the name of the fine-tuning task.")
    parser.add_argument("--task_target_num", type=int, required=False, # ADNI
                        help="Set the number of training samples for the fine-tuning task.")
    parser.add_argument("--stratify", type=str, choices=["strat", "balan"], required=False, # ADNI
                        help="Set training samples are stratified or not.")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)
    
    ### ADNI
    if config.mode == PRETRAINING:
        dataset_train = ADNI_Dataset(config, "no", 0, "no", training=True) # no fine-tuning
        dataset_val = ADNI_Dataset(config, "no", 0, "no", validation=True) # no fine-tuning
        dataset_test = ADNI_Dataset(config, "no", 0, "no", test=True) # no fine-tuning
    elif config.mode == FINE_TUNING:
        dataset_train = ADNI_Dataset(config, args.task_names, args.task_target_num, args.stratify, training=True)
        dataset_val = ADNI_Dataset(config, args.task_names, args.task_target_num, args.stratify, validation=True)
        dataset_test = ADNI_Dataset(config, args.task_names, args.task_target_num, args.stratify, test=True)
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
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, "no", 0, "no") # ADNI
    else:
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_names, args.task_target_num, args.stratify) # ADNI

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        outGT, outPRED = model.fine_tuning() # ADNI
        #print('outGT:', outGT)
        #print('outPRED:', outPRED)
    
    ### ADNI
    if config.mode == FINE_TUNING:
        outGTnp = outGT.cpu().numpy()
        outPREDnp = outPRED.cpu().numpy()
        print('\n<<< Test Results: AUROC >>>')
        outAUROC = []
        for i in range(config.num_classes):
            outAUROC.append(roc_auc_score(outGTnp[:, i], outPREDnp[:, i]))
        aurocMean = np.array(outAUROC).mean()
        print('MEAN', ': {:.4f}'.format(aurocMean))
        
        if 'ADCN' == args.task_names:
            class_names = ['CN', 'AD']
        elif 'MCICN' == args.task_names:
            class_names = ['CN', 'MCI']
        else:
            class_names = ['MCI', 'AD']

        fig, ax = plt.subplots(nrows = 1, ncols = config.num_classes)
        ax = ax.flatten()
        fig.set_size_inches((config.num_classes * 10, 10))
        for i in range(config.num_classes):
            print(class_names[i], ': {:.4f}'.format(outAUROC[i]))
            fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, i], outPRED.cpu()[:, i])
            roc_auc = metrics.auc(fpr, tpr)
            ax[i].plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
            ax[i].set_title('ROC for {0}'.format(class_names[i]))
            ax[i].legend(loc = 'lower right')
            ax[i].plot([0, 1], [0, 1],'r--')
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim([0, 1])
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_xlabel('False Positive Rate')
        
        if config.freeze:
            freezed = 'f'
        else:
            freezed = ''
        plt.savefig('./figs/ADNI_{0}_{1}_{2}{3}_ROC.png'.format(args.task_names, args.stratify, args.task_target_num, freezed), dpi = 100)
        plt.close()
    
    end_time = time.time()
    print('Total', round((end_time - start_time) / 60), 'minutes elapsed.')
    ###
