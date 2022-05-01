import nibabel as nib

my_data = nib.load('brain_to_MNI_nonlin.nii.gz').get_data()
print(my_data.shape) 