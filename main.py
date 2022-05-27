### ADNI
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'
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
import pandas as pd
###

if __name__ == "__main__":
    start_time = time.time() # ADNI
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["pretraining", "finetuning"], required=True,
                        help="Set the training mode. Do not forget to configure config.py accordingly !")
    parser.add_argument("--task_names", type=str, required=False, # ADNI
                        help="Set the name of the fine-tuning task. (e.g. AD/MCI)")
    parser.add_argument("--train_num", type=int, required=True, # ADNI
                        help="Set the number of training samples.")
    parser.add_argument("--stratify", type=str, choices=["strat", "balan"], required=False, # ADNI
                        help="Set training samples are stratified or not for fine-tuning task.")
    args = parser.parse_args()
    mode = PRETRAINING if args.mode == "pretraining" else FINE_TUNING

    config = Config(mode)

    ### ADNI
    if config.mode == PRETRAINING:
        data = pd.read_csv(config.label)
        label_train, label_valid, label_test = np.split(data.sample(frac=1, random_state=1), 
                                                        [args.train_num, int(args.train_num*(1+config.valid_ratio))])
        print('Task: Pretraining')
        print('N = {0}'.format(args.train_num))
        print('meta-data: {0}\n'.format(config.label_name))
        assert len(config.label_name) == len(config.alpha_list), 'len(label_name) and len(alpha_list) should match.'
        assert len(config.label_name) == len(config.label_type), 'len(label_name) and len(label_type) should match.'
        assert len(config.label_name) == len(config.sigma), 'len(alpha_list) and len(sigma) should match.'
        assert sum(config.alpha_list) == 1.0, 'Sum of alpha list should be 1.'

        for i in range(len(config.label_name)): # ["PTAGE", "PTGENDER"]
            if config.label_type[i] != 'cont': # convert str object to numbers
                label_train[config.label_name[i]] = pd.Categorical(label_train[config.label_name[i]])
                label_train[config.label_name[i]] = label_train[config.label_name[i]].cat.codes
                label_valid[config.label_name[i]] = pd.Categorical(label_valid[config.label_name[i]])
                label_valid[config.label_name[i]] = label_valid[config.label_name[i]].cat.codes
                label_test[config.label_name[i]] = pd.Categorical(label_test[config.label_name[i]])
                label_test[config.label_name[i]] = label_test[config.label_name[i]].cat.codes
                
    else: # config.mode == FINE_TUNING:
        labels = pd.read_csv(config.label)
        label_name = config.label_name # 'Dx.new'
        task_include = args.task_names.split('/')
        assert len(task_include) == 2, 'Set only two labels.'

        data_1 = labels[labels[label_name] == task_include[0]]
        data_2 = labels[labels[label_name] == task_include[1]]

        if args.stratify == 'strat':
            ratio = len(data_1) / (len(data_1) + len(data_2))
            len_1_train = round(args.train_num*ratio)
            len_2_train = args.train_num - len_1_train
            len_1_valid = round(int(args.train_num*config.valid_ratio)*ratio)
            len_2_valid = int(args.train_num*config.valid_ratio) - len_1_valid
            train1, valid1, test1 = np.split(data_1.sample(frac=1, random_state=1), 
                                             [len_1_train, len_1_train + len_1_valid])
            train2, valid2, test2 = np.split(data_2.sample(frac=1, random_state=1), 
                                             [len_2_train, len_2_train + len_2_valid])
            label_train = pd.concat([train1, train2]).sample(frac=1)
            label_valid = pd.concat([valid1, valid2]).sample(frac=1)
            label_test = pd.concat([test1, test2]).sample(frac=1)
        else: # args.stratify == 'balan'
            if len(data_1) <= len(data_2):
                limit = len(data_1)
            else:
                limit = len(data_2)
            data_1 = data_1.sample(frac=1, random_state=1)[:limit]
            data_2 = data_2.sample(frac=1, random_state=1)[:limit]
            len_1_train = round(args.train_num*0.5)
            len_2_train = args.train_num - len_1_train
            len_1_valid = round(int(args.train_num*config.valid_ratio)*0.5)
            len_2_valid = int(args.train_num*config.valid_ratio) - len_1_valid
            assert args.train_num*(1+config.valid_ratio) < limit*2, 'Not enough data to make balanced set.'
            train1, valid1, test1 = np.split(data_1.sample(frac=1, random_state=1), 
                                             [len_1_train, len_1_train + len_1_valid])
            train2, valid2, test2 = np.split(data_2.sample(frac=1, random_state=1), 
                                             [len_2_train, len_2_train + len_2_valid])
            label_train = pd.concat([train1, train2]).sample(frac=1)
            label_valid = pd.concat([valid1, valid2]).sample(frac=1)
            label_test = pd.concat([test1, test2]).sample(frac=1)
            assert len(label_test) >= 100, 'Not enough test data. (Total: {0})'.format(len(label_test))

        print('Task: Fine-tuning for {0}'.format(args.task_names))
        print('N = {0}'.format(args.train_num))
        print('Policy: {0}\n'.format(args.stratify))
        print('Train data info:\n{0}\nTotal: {1}\n'.format(label_train[config.label_name].value_counts(sort=False), len(label_train)))
        print('Valid data info:\n{0}\nTotal: {1}\n'.format(label_valid[config.label_name].value_counts(sort=False), len(label_valid)))
        print('Test data info:\n{0}\nTotal: {1}\n'.format(label_test[config.label_name].value_counts(sort=False), len(label_test)))

        label_train[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_valid[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
        label_test[label_name].replace({task_include[0]: 0, task_include[1]: 1}, inplace=True)
    ###
    
    ### ADNI
    if config.mode == PRETRAINING:
        dataset_train = ADNI_Dataset(config, label_train)
        dataset_val = ADNI_Dataset(config, label_valid)
        dataset_test = ADNI_Dataset(config, label_test)
    elif config.mode == FINE_TUNING:
        dataset_train = ADNI_Dataset(config, label_train)
        dataset_val = ADNI_Dataset(config, label_valid)
        dataset_test = ADNI_Dataset(config, label_test)
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
        loss = GeneralizedSupervisedNTXenLoss(config=config, # ADNI
                                              temperature=config.temperature,
                                              sigma=config.sigma,
                                              return_logits=True)
    elif config.mode == FINE_TUNING:
        loss = CrossEntropyLoss()

    if config.mode == PRETRAINING:
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, "no", 0, "no") # ADNI
    else:
        model = yAwareCLModel(net, loss, loader_train, loader_val, loader_test, config, args.task_names, args.train_num, args.stratify) # ADNI

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

        fig, ax = plt.subplots(nrows = 1, ncols = config.num_classes)
        ax = ax.flatten()
        fig.set_size_inches((config.num_classes * 10, 10))
        for i in range(config.num_classes):
            print(task_include[i], ': {:.4f}'.format(outAUROC[i]))
            fpr, tpr, threshold = metrics.roc_curve(outGT.cpu()[:, i], outPRED.cpu()[:, i])
            roc_auc = metrics.auc(fpr, tpr)
            ax[i].plot(fpr, tpr, label = 'AUC = %0.2f' % (roc_auc))
            ax[i].set_title('ROC for {0}'.format(task_include[i]))
            ax[i].legend(loc = 'lower right')
            ax[i].plot([0, 1], [0, 1], 'r--')
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim([0, 1])
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_xlabel('False Positive Rate')
        
        if config.freeze:
            freezed = 'f'
        else:
            freezed = ''
        plt.savefig('./figs/ADNI_{0}_{1}_{2}{3}_ROC.png'.format(args.task_names.replace('/', ''), args.stratify, args.train_num, freezed), dpi = 100)
        plt.close()
    
    end_time = time.time()
    print('Total', round((end_time - start_time) / 60), 'minutes elapsed.')
    ###
