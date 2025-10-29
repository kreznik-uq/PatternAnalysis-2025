import torch.optim as optim
import torch
import torch.nn as nn

from tqdm import tqdm
from modules import UNet3D
from torch.utils.data import DataLoader
from dataset import ProstateDataset
from torch.amp import GradScaler, autocast
from modules import UNet3D
from utils import mean_dice_coefficient, create_patient_aware_split

class DiceLoss(nn.Module):
    """
    Calculates dice loss
    """
    def __init__(self, num_classes, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):

        pred = torch.softmax(pred, dim = 1)
        pred_flat = pred.contiguous().view(pred.shape[0], self.num_classes, -1)
        target_flat = target.contiguous().view(target.shape[0], self.num_classes, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function that is a sum of cross entropy loss and Dice loss
    """
    def __init__(self, num_classes):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)

    def forward(self, pred, targets):   
        target_indices = torch.argmax(targets, dim=1)
        ce = self.cross_entropy(pred, target_indices)
        dice = self.dice(pred,targets)
        return ce * 0.5 + dice * 0.5

# Code reference: https://medium.com/@fernandopalominocobo/
# mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

def validate(model, loader, criterion, device, num_classes):
    """
    Evaluates the model on the evaluation dataset, returning loss and Dice coefficient
    """
    model.eval()
    val_running_loss = 0
    val_running_dc = 0
    
    with torch.no_grad():
        for inputs, labels, image_path in tqdm(loader, desc="Validating"):
            inputs = inputs.float().to(device)
            labels = labels.squeeze(1).permute(0, 4, 1, 2, 3).float().to(device)

            with autocast(device_type="cuda"):
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                
            dc = mean_dice_coefficient(y_pred, labels, num_classes)
            
            val_running_loss += loss.item()
            val_running_dc += dc

    avg_val_loss = val_running_loss / len(loader)
    avg_val_dc = val_running_dc / len(loader)
    
    return avg_val_loss, avg_val_dc

if __name__ == '__main__':
    """
    Main training loop
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ProstateDataset(image_dir="processed_data/images", label_dir="processed_data/labels")
    train_dataset, val_dataset = create_patient_aware_split(dataset, val_split=0.2)

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    validationloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Code reference: https://www.codegenes.net/blog/3d-unet-pytorch/
    model = UNet3D(in_channels=1, out_channels=6).to(device)
    criterion = CombinedLoss(num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1)

    num_epochs = 30
    best_val_dice = 0.0
    model_save_path = "best_model.pth"

    scaler = GradScaler() 

    print("started")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(dataloader)
        for inputs, labels, image_path in progress_bar:
            inputs = inputs.to(device)
            labels = labels.squeeze(1).permute(0, 4, 1, 2, 3).float().to(device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()
            print(f'Epoch {epoch + 1}/{num_epochs}, Step Loss: {loss.item()}')

        #Epoch is over, time to validate
        avg_train_loss = running_loss / len(dataloader)
        avg_val_loss, avg_val_dice = validate(model, validationloader, criterion, device, 6)

        scheduler.step(avg_val_dice)

        print(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model. + {best_val_dice}")

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')