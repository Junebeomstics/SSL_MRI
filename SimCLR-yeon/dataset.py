from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
from torchvision.transforms import transforms
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
    def __init__(self, config, subject_file = './data/fsdat_baseline.csv', mode = 'train', data_dir = './data', training=False, validation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.subject_file = subject_file
        self.mode = mode
        assert training != validation
        df = pd.read_csv(subject_file)
        self.files = list(df['File_name'])
        
        
        #self.transforms = Transformer()
        self.n_views = 2 #assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
        self.config = config
        
        if training:
            #self.data = np.load(config.data_train)
            self.data_dir = data_dir +'/train'
            self.files = [x for x in self.files if x in os.listdir(self.data_dir)]
        elif validation:
            
            self.data_dir = data_dir +'/val'
            self.files = [x for x in self.files if x in os.listdir(self.data_dir)]

        #assert self.data.shape[1:] == tuple(config.input_size), "3D images must have shape {}".\
        #    format(config.input_size)
        

    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y) in list_samples], dim=0)

        return (list_x, list_y)

    def __getitem__(self, idx):

        # For a single input x, samples (t, t') ~ T to generate (t(x), t'(x))
        file = self.files[idx]
        path = os.path.join(self.data_dir, file)
        self.data = nib.load(os.path.join(path,'brain_to_MNI_nonlin.nii.gz')).get_data() #(Assume) self.warped == True #returns array of the image data
        self.data = self.data.reshape(91,109*91)      
        ######################################################################   
     # follow the transform type used in SimCLR    
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        s = 1  
        size = 32 #32 for cifar10, 96 for stl10
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
    ######################################################################    
        x1 = self.data_transforms(self.data)
        x2 = self.data_transforms(self.data)
        labels = self.labels[self.config.label_name].values[idx]
        x = np.stack((x1, x2), axis=0)

        return (x, labels)

    def __len__(self):
        return len(self.files)
        #return len(self.data)
