import torch
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.ndimage import zoom
from modules import UNet3D
from dataset import ProstateDataset
from torch.amp import autocast
from utils import get_case_key, to_channels, calculate_mean_dice_score 

CONFIG = {
    "MODEL_PATH": "best_model.pth",
    "PROCESSED_IMAGE_DIR": "processed_data/images",
    "PROCESSED_LABEL_DIR": "processed_data/labels",
    "ORIGINAL_NIFTI_DIR": "semantic_MRs_anon",
    "ORIGINAL_LABEL_DIR": "semantic_labels_anon",
    "OUTPUT_DIR": "predictions",
    "NUM_CLASSES": 6,
    "BATCH_SIZE": 1,
    "NUM_WORKERS": 8,
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    model = UNet3D(in_channels=1, out_channels=CONFIG["NUM_CLASSES"])
    model.load_state_dict(torch.load(CONFIG["MODEL_PATH"], map_location=device))
    model.to(device)
    model.eval()

    # Setup the dataset and DataLoader for evaluation
    full_dataset = ProstateDataset(
        image_dir=CONFIG["PROCESSED_IMAGE_DIR"],
        label_dir=CONFIG["PROCESSED_LABEL_DIR"]
    )
    dice_scores = []

    eval_loader = DataLoader(
        full_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"]
    )

    print("started")
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs, _, image_paths = batch
            image_tensor = inputs.to(device)

            with autocast(device_type="cuda"):
                pred_logits = model(image_tensor)
            
            # Process each item in the batch
            for i in range(pred_logits.shape[0]):
                single_pred_logit = pred_logits[i].unsqueeze(0)
                pred_mask = torch.argmax(torch.softmax(single_pred_logit, dim=1), dim=1).squeeze(0)
                pred_mask_np = pred_mask.cpu().numpy()

                original_processed_path = image_paths[i]
                base_filename = os.path.basename(original_processed_path).replace('.pt', '.nii.gz')
                original_nifti_path = os.path.join(CONFIG["ORIGINAL_NIFTI_DIR"], base_filename)

                if os.path.exists(original_nifti_path):
                    original_nifti = nib.load(original_nifti_path)
                    original_shape = original_nifti.get_fdata().shape

                    # Resize prediction to original dimensions if necessary
                    if pred_mask_np.shape != original_shape:
                        zoom_factors = [orig_dim / pred_dim for orig_dim, pred_dim in zip(original_shape, pred_mask_np.shape)]
                        full_size_pred_mask = zoom(pred_mask_np, zoom_factors, order=0, mode='nearest')
                    else:
                        full_size_pred_mask = pred_mask_np

                    full_size_pred_mask = full_size_pred_mask.astype(np.uint8)
                    
                    # Construct path to the original label file
                    label_key = get_case_key(base_filename)
                    original_label_filename = base_filename.replace("_LFOV.nii.gz", "_SEMANTIC_LFOV.nii.gz")
                    original_label_path = os.path.join(CONFIG["ORIGINAL_LABEL_DIR"], original_label_filename)

                    original_label_nifti = nib.load(original_label_path)
                    original_label_mask = np.round(original_label_nifti.get_fdata()).astype(np.uint8)
                        
                    original_label_one_hot = to_channels(original_label_mask)
                        
                    score = calculate_mean_dice_score(full_size_pred_mask, original_label_one_hot, CONFIG["NUM_CLASSES"])
                    dice_scores.append(score)

                    # Save the prediction as a NIfTI file
                    pred_nifti = nib.Nifti1Image(full_size_pred_mask, affine=original_nifti.affine, header=original_nifti.header)
                    output_path = os.path.join(CONFIG["OUTPUT_DIR"], base_filename.replace('.nii.gz', '_pred.nii.gz'))
                    nib.save(pred_nifti, output_path)
                else:
                    print(f"Warning: Could not find original NIfTI file at {original_nifti_path} to save prediction.")

    # Calculate and print the average Dice score and standard deviation
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
        
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Standard Deviation of Dice Scores: {std_dice:.4f}")

if __name__ == '__main__':
    main()