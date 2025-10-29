import numpy as np
import os
import torch.nn as nn
import torch
import re
from torch.utils.data import Subset

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    arr = arr.astype(np.int64) 
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)

    for i, c in enumerate(channels):
        res[..., i][arr == c] = 1

    return res

def calculate_mean_dice_score(pred_mask, target_one_hot, num_classes, smooth=1e-6):
    """Calculates the mean Dice score, excluding the background class."""
    dice_per_class = []
    # Start from class 1 to exclude the background
    for i in range(1, num_classes):
        pred_class = (pred_mask == i).astype(np.float32)
        target_class = target_one_hot[..., i].astype(np.float32)

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)

    return np.mean(dice_per_class) if dice_per_class else 0.0

def numpy_to_one_hot(mask, num_classes):
    """Converts a numpy mask to a one-hot encoded array."""
    mask = mask.astype(np.int64)
    shape = mask.shape
    one_hot = np.zeros(shape + (num_classes,), dtype=np.uint8)

    for i in range(num_classes):
        one_hot[..., i][mask == i] = 1
    return one_hot

def get_case_key(filename: str) -> str:
    """Extracts a case key from a filename."""
    match = re.search(r'Case_(\d+)', os.path.basename(filename))
    return match.group(1)

# Code reference: https://medium.com/data-scientists-diary/
# implementation-of-dice-loss-vision-pytorch-7eef1e438f68

def mean_dice_coefficient(pred, target, num_classes, smooth=1e-6):
    pred_mask = torch.argmax(pred, dim=1)
    
    pred_one_hot = nn.functional.one_hot(pred_mask, num_classes).permute(0, 4, 1, 2, 3)

    dice_per_class = []
    for i in range(1, num_classes):
        pred_class = pred_one_hot[:, i, :, :, :]
        target_class = target[:, i, :, :, :]
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice.item())
        
    return sum(dice_per_class) / len(dice_per_class) if dice_per_class else 0.0

def get_patient_id_from_path(filepath: str) -> str:
    filename = os.path.basename(filepath)
    match = re.search(r'Case_(\d+)', filename)
    return match.group(1)


def create_patient_aware_split(dataset, val_split=0.2):
    """
    Splits a dataset into training and validation sets, ensuring that all
    images from a single patient belong to only one set.
    """

    all_image_paths = dataset.image_paths
    patient_ids = [get_patient_id_from_path(p) for p in all_image_paths]
    unique_patients = np.unique(patient_ids)

    # Shuffle the unique patient IDs
    np.random.shuffle(unique_patients)

    # Split patient IDs into training and validation sets
    val_num_patients = int(len(unique_patients) * val_split)
    train_num_patients = len(unique_patients) - val_num_patients
    
    train_patient_ids = unique_patients[:train_num_patients]
    val_patient_ids = unique_patients[train_num_patients:]
    
    # Create lists of indices for the full dataset
    train_indices = []
    val_indices = []
    
    for idx, patient_id in enumerate(patient_ids):
        if patient_id in train_patient_ids:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
            
    print(f"Splitting into {len(train_patient_ids)} training patients and {len(val_patient_ids)} validation patients.")
    print(f"Resulting in {len(train_indices)} training images and {len(val_indices)} validation images.")

    # Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset