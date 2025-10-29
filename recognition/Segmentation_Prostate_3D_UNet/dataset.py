import numpy as np
import nibabel as nib
import torch
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from utils import to_channels, get_case_key

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
getAffines=False, orient=False, early_stop=False, downsample_factor=None, interpolation_order=3):
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

    niftiImage = nib.load(imageNames)
    if orient:
        niftiImage = nib.as_closest_canonical(niftiImage)

    first_case = niftiImage.get_fdata()
    affine = niftiImage.affine 

    if not normImage:
        first_case = np.round(first_case).astype(np.int16)

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0] # sometimes extra dims, remove
    if downsample_factor:
        first_case = zoom(first_case, downsample_factor, order=interpolation_order)

    if normImage:
        mean = np.mean(first_case)
        std = np.std(first_case)
        if std > 0:
            first_case = (first_case - mean) / std

    if categorical:
        first_case = to_channels(first_case, dtype=np.uint8)

    if getAffines:
        return first_case.astype(dtype), affine
    else:
        return first_case.astype(dtype)

class ProstateDataset(Dataset):
    def __init__(self, image_dir, label_dir, downsample_factor=0.5):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.pt')))
        self.label_paths = sorted(glob.glob(os.path.join(label_dir, '*.pt')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_tensor = torch.load(self.image_paths[idx])
        label_tensor = torch.load(self.label_paths[idx])

        return image_tensor, label_tensor, self.image_paths

class ProstateDatasetEvaluate(Dataset):

    def __init__(self, image_dir, label_dir, downsample_factor=0.5):
        image_map = {get_case_key(p): p for p in glob.glob(os.path.join(image_dir, '*.pt'))}
        label_map = {get_case_key(p): p for p in glob.glob(os.path.join(label_dir, '*.pt'))}

        image_keys = set(image_map.keys())
        label_keys = set(label_map.keys())
        common_keys = sorted(list(image_keys.intersection(label_keys)))
        self.file_pairs = [{'image': image_map[key], 'label': label_map[key]} for key in common_keys]

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):

        pair = self.file_pairs[idx]
        
        image_path = pair['image']
        label_path = pair['label']
        
        image_tensor = torch.load(image_path)
        label_tensor = torch.load(label_path)

        return image_tensor, label_tensor, image_path

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
    IMAGE_DIR = "semantic_MRs_anon"
    LABEL_DIR = "semantic_labels_anon"

    PROCESSED_IMAGE_DIR = "processed_data/images"
    PROCESSED_LABEL_DIR = "processed_data/labels"
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_LABEL_DIR, exist_ok=True)

    TARGET_SHAPE = (128, 128, 64)
    DOWNSAMPLE_FACTOR = 0.5

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, '*.nii.gz')))
    label_paths = sorted(glob.glob(os.path.join(LABEL_DIR, '*.nii.gz')))

    for i, (img_path, lbl_path) in enumerate(tqdm(zip(image_paths, label_paths), total=len(image_paths))):
        image_array = load_data_3D(img_path, normImage=True, categorical=False, downsample_factor=DOWNSAMPLE_FACTOR, dtype=np.float32)
        label_array = load_data_3D(lbl_path, normImage=False,categorical=False, downsample_factor=DOWNSAMPLE_FACTOR, dtype=np.uint8, interpolation_order=0)

        image_array = resample_or_pad_volume(image_array, target_shape=TARGET_SHAPE)
        label_array = resample_or_pad_volume(label_array, target_shape=TARGET_SHAPE)

        label_one_hot = to_channels(label_array, dtype=np.uint8)

        image_tensor = torch.from_numpy(image_array).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label_one_hot.copy()).to(torch.uint8)

        base_image_filename = os.path.basename(img_path).replace('.nii.gz', '.pt')
        base_label_filename = os.path.basename(lbl_path).replace('.nii.gz', '.pt')
        torch.save(image_tensor, os.path.join(PROCESSED_IMAGE_DIR, base_image_filename))
        torch.save(label_tensor, os.path.join(PROCESSED_LABEL_DIR, base_label_filename))

if __name__ == '__main__':
    preprocess()