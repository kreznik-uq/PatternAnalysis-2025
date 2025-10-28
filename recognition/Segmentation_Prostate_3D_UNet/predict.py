import torch
import numpy as np
import nibabel as nib
import os
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast
from monai.inferers import sliding_window_inference
from scipy.ndimage import zoom
from modules import UNet3D
from dataset import ProstateDatasetEvaluate

CONFIG = {
    "MODEL_PATH": "best_model.pth",
    "PROCESSED_IMAGE_DIR": "processed_data/images",
    "PROCESSED_LABEL_DIR": "processed_data/labels",
    "ORIGINAL_NIFTI_DIR": "semantic_MRs_anon", 
    "ORIGINAL_LABEL_DIR": "semantic_labels_anon",
    "OUTPUT_DIR": "predictions",
    "NUM_CLASSES": 6,
    "ROI_SIZE": (128, 128, 64),
    "SW_BATCH_SIZE": 4,
    "OVERLAP": 0.5,
    "VALIDATION_SPLIT": 0.2,
    "RANDOM_SEED": 42,
    "NUM_WORKERS": 8,
}

def dice_coefficient(pred, target):
    pred_probs = torch.sigmoid(pred)
    pred_mask = (pred_probs > 0.5).float()
    intersection = (pred_mask * target).sum()
    union = pred_mask.sum() + target.sum()
    
    dice = (2. * intersection) / (union + 1e-6)
    return dice.item()

def calculate_mean_dice_score(pred_mask, target_one_hot, num_classes, smooth=1e-6):
    dice_per_class = []
    for i in range(1, num_classes):
        pred_class = (pred_mask == i).astype(np.float32)
        target_class = target_one_hot[..., i].astype(np.float32)
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)
    return np.mean(dice_per_class) if dice_per_class else 0.0

def numpy_to_one_hot(mask, num_classes):
    mask = mask.astype(np.int64)
    
    shape = mask.shape
    one_hot = np.zeros(shape + (num_classes,), dtype=np.uint8)
    
    for i in range(num_classes):
        one_hot[..., i][mask == i] = 1
    return one_hot

def get_case_key(filename: str) -> str:
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return ""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    model = UNet3D(in_channels=1, out_channels=CONFIG["NUM_CLASSES"])
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
    model.to(device)
    model.eval()

    full_dataset = ProstateDatasetEvaluate(
        image_dir=CONFIG["PROCESSED_IMAGE_DIR"], 
        label_dir=CONFIG["PROCESSED_LABEL_DIR"]
    )
    
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    _, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    validation_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=CONFIG["NUM_CLASSES"])
    
    dice_scores = []
    
    with torch.no_grad():
        for inputs, labels, image_paths in tqdm(validation_loader, desc="Evaluating"):
            image_tensor = inputs.to(device)
            original_processed_path = image_paths[0]

            def model_amp_predictor(data):
                with autocast(device_type="cuda"):
                    return model(data)

            pred_logits = sliding_window_inference(
                inputs=image_tensor,
                roi_size=CONFIG["ROI_SIZE"],
                sw_batch_size=4,
                predictor=model_amp_predictor,
                overlap=0.5,
                mode="gaussian"
            )
            
            pred_mask = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1).squeeze(0)
            pred_mask_np = pred_mask.cpu().numpy()
            
            base_filename = os.path.basename(original_processed_path).replace('.pt', '.nii.gz')
            original_nifti_path = os.path.join(CONFIG["ORIGINAL_NIFTI_DIR"], base_filename)

            if os.path.exists(original_nifti_path):
                original_nifti = nib.load(original_nifti_path)
                original_shape = original_nifti.get_fdata().shape

                if pred_mask_np.shape != original_shape:
                    zoom_factors = [orig_dim / pred_dim for orig_dim, pred_dim in zip(original_shape, pred_mask_np.shape)]
                    full_size_pred_mask = zoom(pred_mask_np, zoom_factors, order=0, mode='nearest')
                else:
                    full_size_pred_mask = pred_mask_np
                
                full_size_pred_mask = full_size_pred_mask.astype(np.uint8)

                label_key = get_case_key(base_filename)
                original_label_filename = base_filename.replace(label_key, f"{label_key}_SEMANTIC") if label_key in base_filename else f"{label_key}_SEMANTIC.nii.gz"
                original_label_path = os.path.join("semantic_labels_anon/", original_label_filename)
                
                if os.path.exists(original_label_path):
                     original_label_nifti = nib.load(original_label_path)
                     original_label_mask = np.round(original_label_nifti.get_fdata()).astype(np.uint8)
                     
                     original_label_one_hot = numpy_to_one_hot(original_label_mask, num_classes=CONFIG["NUM_CLASSES"])
                     
                     score = calculate_mean_dice_score(full_size_pred_mask, original_label_one_hot, CONFIG["NUM_CLASSES"])
                     dice_scores.append(score)
                
                pred_nifti = nib.Nifti1Image(full_size_pred_mask, affine=original_nifti.affine, header=original_nifti.header)
                output_path = os.path.join(CONFIG["OUTPUT_DIR"], base_filename.replace('.nii.gz', '_pred.nii.gz'))
                nib.save(pred_nifti, output_path)
            else:
                print(f"Warning: Could not find original NIfTI file at {original_nifti_path} to save prediction.")


    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Standard Deviation: {std_dice:.4f}")

if __name__ == '__main__':
    main()