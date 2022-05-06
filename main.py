import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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
    parser.add_argument("--target_num", type=int, required=True, # ADNI
                        help="Set the number of training samples.")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)

    if config.mode == mode:
        ### ADNI
        dataset_train = MRIDataset(config, args.target_num, training=True)
        dataset_val = MRIDataset(config, args.target_num, validation=True)
        dataset_test = MRIDataset(config, args.target_num, test=True)
        ###
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

    model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config) # ADNI

    if config.mode == PRETRAINING:
        model.pretraining()
    else:
        outGT, outPRED = model.fine_tuning() # ADNI
    
    ### ADNI
    outAUROC = []
    outGTnp = outGT.cpu().numpy()
    outPREDnp = outPRED.cpu().numpy()
    for i in range(config.num_classes):
        outAUROC.append(roc_auc_score(outGTnp[:, i], outPREDnp[:, i]))
    
    class_names = ['CN', 'MCI', 'AD']
    aurocMean = np.array(outAUROC).mean()
    print('<<< Model Test Results: AUROC >>>')
    print('MEAN', ': {:.4f}\n'.format(aurocMean))
    for i in range (0, len(outAUROC)):
        print(class_names[i], ': {:.4f}'.format(outAUROC[i]))
    
    fig, ax = plt.subplots(nrows = 1, ncols = config.num_classes)
    ax = ax.flatten()
    fig.set_size_inches((config.num_classes * 10, 10))
    for i in range(config.num_classes):
        fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, i], outPRED.cpu()
        [:, i])
        roc_auc = metrics.auc(fpr, tpr)
        
        ax[i].plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
        ax[i].set_title('ROC for: ' + class_names[i])
        ax[i].legend(loc = 'lower right')
        ax[i].plot([0, 1], [0, 1],'r--')
        ax[i].set_xlim([0, 1])
        ax[i].set_ylim([0, 1])
        ax[i].set_ylabel('True Positive Rate')
        ax[i].set_xlabel('False Positive Rate')

    plt.savefig('./ADNI_ROC_100.png', dpi = 100)
    plt.close()
    ###
