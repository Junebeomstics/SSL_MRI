from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
from PIL import Image
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip
from data_aug.gaussian_blur import GaussianBlur
import nibabel as nib
#from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class MRIDataset(Dataset):
    #change > Class from the root_folder
    def __init__(self, config, mode = 'train', subject_file = './data/fsdat_baseline.csv', data_dir = './data', training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.train_subject_folder = os.path.join(self.data_dir, 'fsdat_baseline_train.csv')
        self.val_subject_folder = os.path.join(self.data_dir, 'fsdat_baseline_val.csv')
        self.subject_file = subject_file
        self.mode = mode
        df = pd.read_csv(subject_file)
        self.files = list(df['File_name'])
        
        assert training != validation
        
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

        #self.transforms = Transformer()
        self.n_views = 2 #assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
        
        if training:
            #self.data = np.load(config.data_train)
            df = pd.read_csv(self.train_subject_folder)
            self.data_dir = data_dir +'/train'
            self.files = [x for x in self.files if x in os.listdir(self.data_dir)]
            self.data = []
            for f in range(len(self.files)):
                self.data.append(os.path.join(self.data_dir, self.files[f]))
            
        elif validation:
            df = pd.read_csv(self.val_subject_folder)
            #self.files = list(df['File_name']) #list of files
            self.data_dir = data_dir +'/val'
            self.files = [x for x in self.files if x in os.listdir(self.data_dir)]

            
            self.data = []
            for f in range(len(self.files)):
                self.data.append(os.path.join(self.data_dir, self.files[f]))
            
            

        #assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".\
        #    format(config.input_size)
        

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
     # follow the transform type used in SimCLR    
        img = nib.load(os.path.join(self.data[idx],'brain_to_MNI_nonlin.nii.gz')).get_data() #(Assume) self.warped == True #returns array of the image data
        x1 = self.transforms(img)
        x2 = self.transforms(img)
        labels = self.labels[self.config.label_name].values[idx]
        x = np.stack((x1, x2), axis=0)

        return (x, labels)

    def __len__(self):
        return len(self.files)
