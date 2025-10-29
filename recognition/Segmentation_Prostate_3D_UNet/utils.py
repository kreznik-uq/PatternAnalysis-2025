import numpy as np
import os
import torch.nn as nn
import torch

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
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return ""

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