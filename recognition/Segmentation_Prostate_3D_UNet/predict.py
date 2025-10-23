import torch
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast
from monai.inferers import sliding_window_inference

from modules import UNet3D
from dataset import ProstateDataset

MODEL_PATH = "best_model.pth"
PROCESSED_IMAGE_DIR = "processed_data/images"
PROCESSED_LABEL_DIR = "processed_data/labels"
OUTPUT_DIR = "validation_outputs_monai_amp"
ORIGINAL_NIFTI_DIR = "semantic_MRs_anon"

ROI_SIZE = (96, 96, 32)
SW_BATCH_SIZE = 4
OVERLAP = 0.5

def dice_coefficient(pred, target):
    pred_probs = torch.sigmoid(pred)
    pred_mask = (pred_probs > 0.5).float()
    intersection = (pred_mask * target).sum()
    union = pred_mask.sum() + target.sum()
    
    dice = (2. * intersection) / (union + 1e-6)
    return dice.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = UNet3D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    full_dataset = ProstateDataset(image_dir=PROCESSED_IMAGE_DIR, label_dir=PROCESSED_LABEL_DIR)
    
    torch.manual_seed(42)
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    validation_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    dice_scores = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(validation_loader, desc="Evaluating")):
            image_tensor = inputs.to(device)
            label_array = labels.squeeze().numpy()

            def model_amp_predictor(data):
                with autocast(device_type="cuda"):
                    return model(data)

            pred_logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=model_amp_predictor,
                overlap=OVERLAP,
                mode="gaussian"
            )
            
            pred_probs = torch.sigmoid(pred_logits).squeeze().cpu().numpy()
            
            score = dice_coefficient(pred_probs, label_array)
            dice_scores.append(score)

    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Standard Deviation: {std_dice:.4f}")

if __name__ == '__main__':
    main()