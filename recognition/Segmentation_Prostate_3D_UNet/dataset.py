import numpy as np
import nibabel as nib
import torch
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.ndimage import zoom

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
getAffines=False, orient=False, early_stop=False, downsample_factor=None):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory
usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice
thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is
labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick
loading and testing scripts.
    '''
    affines = []

    # ~ interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8: # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = nib.as_closest_canonical(niftiImage)
    # ~ testResultName = "oriented.nii.gz"
    # ~ niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0] # sometimes extra dims, remove
    if downsample_factor:
        first_case = zoom(first_case, downsample_factor, order=0 if categorical else 3)
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = nib.as_closest_canonical(niftiImage)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0] # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth] # clip slices
        if downsample_factor:
            inImage = zoom(inImage, downsample_factor, order=0 if categorical else 3)
        inImage = inImage.astype(dtype)
        if normImage:
            # ~ inImage = inImage / np.linalg.norm(inImage)
            # ~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            # ~ images[i, :, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage # with pad
        else:
            # ~ images[i, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


class ProstateDataset(Dataset):

    def __init__(self, image_dir, label_dir, downsample_factor=0.5):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.nii.gz')))
        self.downsample_factor = downsample_factor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_data_3D([self.image_paths[idx]], normImage=True, downsample_factor=self.downsample_factor)[0]
        label = load_data_3D([self.label_paths[idx]], dtype=np.uint8, categorical=False, downsample_factor=self.downsample_factor)[0]

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label).float().unsqueeze(0)

        return image_tensor, label_tensor