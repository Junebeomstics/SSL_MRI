from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip
### ADNI
import os
import nibabel as nib
from skimage.transform import resize

class ADNI_Dataset(Dataset):

    def __init__(self, config, task_name, task_target_num, stratify, training=False, validation=False, test=False, *args, **kwargs): # ADNI
        super().__init__(*args, **kwargs)
        ### ADNI
        if training:
            assert training != validation
            assert training != test
        elif validation:
            assert training != validation
            assert validation != test
        else:
            assert test != validation
            assert training != test
        ###

        self.transforms = Transformer()
        self.config = config
        self.transforms.register(Normalize(), probability=1.0)
        
        if config.tf == "all_tf":
            self.transforms.register(Flip(), probability=0.5)
            self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=0.5)
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=0.5)

        elif config.tf == "cutout":
            self.transforms.register(Cutout(patch_size=np.ceil(np.array(config.input_size)/4)), probability=1)

        elif config.tf == "crop":
            self.transforms.register(Crop(np.ceil(0.75*np.array(config.input_size)), "random", resize=True),
                                     probability=1)
        
        ### ADNI
        if config.mode == 0: # Define pre-training dataset
            if training:
                self.data_dir = './adni_t1s_baseline'
                self.labels = pd.read_csv('./csv/CN_train.csv')
                self.files = [x for x in os.listdir(self.data_dir) if x[4:12] in list(self.labels['SubjectID'])]
                
            elif validation:
                self.data_dir = './adni_t1s_baseline'
                self.labels = pd.read_csv('./csv/CN_valid.csv')
                self.files = [x for x in os.listdir(self.data_dir) if x[4:12] in list(self.labels['SubjectID'])]

        else: # Define fine-tuning dataset
            if training:
                self.data_dir = './adni_t1s_baseline'
                self.labels = pd.read_csv('./csv/{0}CN_{1}_train{2}.csv'.format(task_name, stratify, task_target_num))                
                self.files = [x for x in os.listdir(self.data_dir) if x[4:12] in list(self.labels['SubjectID'])]
                #self.data = np.load(config.data_train)

            elif validation:
                self.data_dir = './adni_t1s_baseline'
                self.labels = pd.read_csv('./csv/{0}CN_{1}_valid{2}.csv'.format(task_name, stratify, task_target_num))
                self.files = [x for x in os.listdir(self.data_dir) if x[4:12] in list(self.labels['SubjectID'])]
                #self.data = np.load(config.data_val)
                
            elif test:
                self.data_dir = './adni_t1s_baseline'
                self.labels = pd.read_csv('./csv/{0}CN_{1}_test{2}.csv'.format(task_name, stratify, task_target_num))
                self.files = [x for x in os.listdir(self.data_dir) if x[4:12] in list(self.labels['SubjectID'])]
                #self.data = np.load(config.data_val)
            
        #assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".\
        #    format(config.input_size)
        ###
        

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
        ### ADNI
        if self.config.mode == 0: # Pre-training
            label = self.labels[self.config.label_name].values[idx]
            labels = float(label)
        else: # Fine-tuning
            label = self.labels['Dx.new'].values[idx]
            if label == 'CN':
                labels = torch.LongTensor([0])
            else:
                labels = torch.LongTensor([1])
        SubjectID = self.labels['SubjectID'].values[idx]
        file_match = [file for file in self.files if SubjectID in file]
        path = os.path.join(self.data_dir, file_match[0])
        img = nib.load(os.path.join(path, 'brain_to_MNI_nonlin.nii.gz'))
        img = np.swapaxes(img.get_data(),1,2)
        img = np.flip(img,1)
        img = np.flip(img,2)
        img = resize(img, (self.config.input_size[1], self.config.input_size[2], self.config.input_size[3]), mode='constant')
        img = torch.from_numpy(img).float().view(self.config.input_size[0], self.config.input_size[1], self.config.input_size[2], self.config.input_size[3])
        img = img.numpy()
        
        np.random.seed()

        if self.config.mode == 0: # Pre-training
            x1 = self.transforms(img)
            x2 = self.transforms(img)
            x = np.stack((x1, x2), axis=0)
        else: # Fine-tuning
            x = self.transforms(img)
        ###
        
        return (x, labels)

    def __len__(self):
        return len(self.labels)
