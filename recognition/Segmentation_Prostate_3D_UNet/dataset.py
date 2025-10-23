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
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.pt')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.pt')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = torch.load(self.image_paths[idx])
        label_tensor = torch.load(self.label_paths[idx])

        return image_tensor, label_tensor

def resample_or_pad_volume(volume, target_shape=(128, 128, 64)):

    current_shape = volume.shape
    target_h, target_w, target_d = target_shape
    h_diff = target_h - current_shape[0]
    if h_diff > 0:
        pad_before, pad_after = h_diff // 2, h_diff - (h_diff // 2)
        volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
    elif h_diff < 0:
        crop_start = abs(h_diff) // 2
        volume = volume[crop_start:crop_start + target_h, :, :]
    w_diff = target_w - current_shape[1]
    if w_diff > 0:
        pad_before, pad_after = w_diff // 2, w_diff - (w_diff // 2)
        volume = np.pad(volume, ((0, 0), (pad_before, pad_after), (0, 0)), mode='constant')
    elif w_diff < 0:
        crop_start = abs(w_diff) // 2
        volume = volume[:, crop_start:crop_start + target_w, :]
    d_diff = target_d - current_shape[2]
    if d_diff > 0:
        pad_before, pad_after = d_diff // 2, d_diff - (d_diff // 2)
        volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), mode='constant')
    elif d_diff < 0:
        crop_start = abs(d_diff) // 2
        volume = volume[:, :, crop_start:crop_start + target_d]
    return volume

def preprocess():
    IMAGE_DIR = "semantic_labels_anon"
    LABEL_DIR = "semantic_MRs_anon"

    PROCESSED_IMAGE_DIR = "processed_data/images"
    PROCESSED_LABEL_DIR = "processed_data/labels"
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_LABEL_DIR, exist_ok=True)

    TARGET_SHAPE = (128, 128, 64)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.nii.gz')))
    label_paths = sorted(glob.glob(os.path.join(LABEL_DIR, '*.nii.gz')))

    for i, (img_path, lbl_path) in enumerate(tqdm(zip(image_paths, label_paths), total=len(image_paths))):
        image_array = load_data_3D([img_path], normImage=True, is_label=False)[0]
        label_array = load_data_3D([lbl_path], normImage=False,is_label=True)[0]

        image_array = resample_or_pad_volume(image_array, target_shape=TARGET_SHAPE)
        label_array = resample_or_pad_volume(label_array, target_shape=TARGET_SHAPE)

        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label_array).float().unsqueeze(0)

        base_filename = os.path.basename(img_path).replace('.nii.gz', '.pt')
        torch.save(image_tensor, os.path.join(PROCESSED_IMAGE_DIR, base_filename))
        torch.save(label_tensor, os.path.join(PROCESSED_LABEL_DIR, base_filename))

if __name__ == '__main__':
    preprocess()