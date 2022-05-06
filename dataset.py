from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip
### ADNI
import os
import nibabel as nib
from skimage.transform import resize

class MRIDataset(Dataset):

    def __init__(self, config, target_num, training=False, validation=False, test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if training:
            assert training != validation
            assert training != test
        elif validation:
            assert training != validation
            assert validation != test
        else:
            assert test != validation
            assert training != test

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
        if training:
            self.data_dir = './adni_t1s_baseline'
            self.files = [x for x in os.listdir(self.data_dir) if x in os.listdir(self.data_dir)]
            #self.data = np.load(config.data_train)
            self.labels = pd.read_csv('./csv/fsdat_baseline_train_{0}.csv'.format(target_num))
        elif validation:
            self.data_dir = './adni_t1s_baseline'
            self.files = [x for x in os.listdir(self.data_dir) if x in os.listdir(self.data_dir)]
            #self.data = np.load(config.data_val)
            self.labels = pd.read_csv('./csv/fsdat_baseline_valid1_{0}.csv'.format(target_num)) # doesn't matter
        elif test:
            self.data_dir = './adni_t1s_baseline'
            self.files = [x for x in os.listdir(self.data_dir) if x in os.listdir(self.data_dir)]
            #self.data = np.load(config.data_val)
            self.labels = pd.read_csv('./csv/fsdat_baseline_test_{0}.csv'.format(target_num))
        
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
        file = self.files[idx]
        path = os.path.join(self.data_dir, file)
        img = nib.load(os.path.join(path, 'brain_to_MNI_nonlin.nii.gz'))
        img = np.swapaxes(img.get_data(),1,2)
        img = np.flip(img,1)
        img = np.flip(img,2)
        img = resize(img, (121, 145, 121), mode='constant')
        img = torch.from_numpy(img).float().view(1, 121, 145, 121)
        img = img.numpy()
        
        np.random.seed()
        #x1 = self.transforms(img)
        #x2 = self.transforms(img)
        x = self.transforms(img)
        label = self.labels['Dx.new'].values[idx]
        if label == 'CN':
            labels = torch.LongTensor([0])
        elif label == 'MCI':
            labels = torch.LongTensor([1])
        elif label == 'AD':
            labels = torch.LongTensor([2])
        #x = np.stack((x1, x2), axis=0)
        ###
        
        return (x, labels)

    def __len__(self):
        return len(self.labels)
