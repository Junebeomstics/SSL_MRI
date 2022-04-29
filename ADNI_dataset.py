import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
import monai
from monai.transforms import ScaleIntensity

class ADNIdataset(Dataset):
    def __init__(self, subject_file='.',data_dir='../ADNI', normalization=False,augmentation=False, cn_only=True, warped=False, img_size=64):
        self.subject_file = subject_file
        self.data_dir = data_dir
        self.normalization = normalization
        self.augmentation = augmentation
        self.img_size = img_size
        self.warped = warped
        # self.files = os.listdir(self.data_dir)
        df = pd.read_csv(subject_file)
        
        if cn_only ==True:
            df = df[df['Dx.new']=='CN']
        self.files = list(df['File_name'])
        if self.warped == True:
            self.files = [x for x in self.files if x in os.listdir(self.data_dir)]
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):   
        file = self.files[index]
        path = os.path.join(self.data_dir,file)
        if self.warped == True:
            img = nib.load(os.path.join(path,'brain_to_MNI_nonlin.nii.gz'))
        else:
            img = nib.load(os.path.join(path,'brain.mgz'))
        #path = os.path.join(path,rname,aname,'mri')
        #img = nib.load(os.path.join(path,'image.nii'))
        img = np.swapaxes(img.get_data(),1,2)
        img = np.flip(img,1)
        img = np.flip(img,2)

#         true_points = np.argwhere(img)
#         top_left = true_points.min(axis=0)
#         bottom_right = true_points.max(axis=0)
#         img = img[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1, top_left[2]:bottom_right[2] + 1]
        img = resize(img, (self.img_size,self.img_size,self.img_size), mode='constant')
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3*torch.rand(1)[0]+0.7
            if random_n[0] > 0.5:
                img = np.flip(img,0)

            img = img*random_i.data.cpu().numpy()
           
        if self.normalization:
            transform = ScaleIntensity(minv=-1.0, maxv=1.0) #[-1,1] MinMax scaler
            img = transform(img)
        imageout = torch.from_numpy(img).float().view(1,self.img_size,self.img_size,self.img_size)
#         imageout = imageout*2-1


        return imageout

class ADNIdatasetEMB(Dataset):
    def __init__(self, subject_file='.',data_dir='../ADNI', normalization=False,augmentation=False, cn_only=True, warped=False, img_size=64):
        self.subject_file = subject_file
        self.data_dir = data_dir
        self.normalization = normalization
        self.augmentation = augmentation
        self.img_size = img_size
        self.warped = warped
        # self.files = os.listdir(self.data_dir)
        df = pd.read_csv(subject_file)
        
        if cn_only ==True:
            df = df[df['Dx.new']=='CN']
            
        self.files = list(df['File_name'])
        self.sex = list(df['PTGENDER'])
        self.age = list(df['PTAGE']) ## Age min max 55~95
        
        if self.warped == True:
            files = []
            sexes = []
            ages = []
            for file,sex,age in zip(self.files,self.sex,self.age):
                if file in os.listdir(self.data_dir):
                    files.append(file)
                    sexes.append(sex)
                    ages.append(age)
            self.files = files
            self.age = ages
            self.sex = sexes
        self.sex = [0 if x =='M' else 1 for x in self.sex]
       
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):   
        file = self.files[index]
        path = os.path.join(self.data_dir,file)
        if self.warped == True:
            img = nib.load(os.path.join(path,'brain_to_MNI_nonlin.nii.gz'))
        else:
            img = nib.load(os.path.join(path,'brain.mgz'))

        #path = os.path.join(path,rname,aname,'mri')
        #img = nib.load(os.path.join(path,'image.nii'))
        img = np.swapaxes(img.get_data(),1,2)
        img = np.flip(img,1)
        img = np.flip(img,2)
        

#        true_points = np.argwhere(img)
#        top_left = true_points.min(axis=0)
#        bottom_right = true_points.max(axis=0)
#        img = img[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1, top_left[2]:bottom_right[2] + 1]
        img = resize(img, (self.img_size,self.img_size,self.img_size), mode='constant')
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3*torch.rand(1)[0]+0.7
            if random_n[0] > 0.5:
                img = np.flip(img,0)

            img = img*random_i.data.cpu().numpy()
           
        if self.normalization:
            transform = ScaleIntensity(minv=-1.0, maxv=1.0) #[-1,1] MinMax scaler
            img = transform(img)
        imageout = torch.from_numpy(img).float().view(1,self.img_size,self.img_size,self.img_size)
#         imageout = imageout*2-1

        sex = self.sex[index]
        sex = torch.from_numpy(np.array(sex)).float()

        #min max 55 95
        age = self.age[index]
        age_max = np.max(self.age)
        age_min = np.min(self.age)
        age_minmax = (np.array(age)-age_min)/(age_max-age_min)
        #age_minmax = torch.from_numpy(age_minmax).float()
    

        return imageout, np.expand_dims(sex, axis=0), np.expand_dims(age_minmax, axis=0)
    

