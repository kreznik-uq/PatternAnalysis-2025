import torch.optim as optim
import torch
import tqdm
import torch.nn as nn
from modules import UNet3D
from torch.utils.data import DataLoader
from dataset import ProstateDataset
from torch.amp import GradScaler, autocast

# Code reference: https://medium.com/data-scientists-diary/
# implementation-of-dice-loss-vision-pytorch-7eef1e438f68

def dice_coefficient(pred, target):
    pred_probs = torch.sigmoid(pred)
    pred_mask = (pred_probs > 0.5).float()
    intersection = (pred_mask * target).sum()
    union = pred_mask.sum() + target.sum()
    
    dice = (2. * intersection) / (union + 1e-6)
    return dice.item()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, smooth=1):

        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2. * intersection + smooth) / (union + smooth)

        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, targets):   
        bce = self.bce(pred,targets)
        dice = self.dice(pred,targets)
        return bce * 0.5 + dice * 0.5

# Code reference: https://medium.com/@fernandopalominocobo/
# mastering-u-net-a-step-by-step-guide-to-segmentation-from-scratch-with-pytorch-6a17c5916114

def validate(model, loader, criterion, device):
    model.eval()
    val_running_loss = 0
    val_running_dc = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader, desc="Validating"):
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            if labels.max() > 1.0:
                labels = labels / 255.0

            with autocast(device_type="cuda"):
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                
            dc = dice_coefficient(y_pred, labels)
            
            val_running_loss += loss.item()
            val_running_dc += dc

    avg_val_loss = val_running_loss / len(loader)
    avg_val_dc = val_running_dc / len(loader)
    
    return avg_val_loss, avg_val_dc

if __name__ == '__main__':

    dataset = ProstateDataset(image_dir="processed_data/images", label_dir="processed_data/labels")

    dataset_size = len(dataset)
    val_size = int(dataset_size * 0.2)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    validationloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # Code reference: https://www.codegenes.net/blog/3d-unet-pytorch/
    model = UNet3D(in_channels=1, out_channels=1)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1)

    num_epochs = 20
    best_val_dice = 0.0
    model_save_path = "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler() 

    print("started")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm.tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            labels = labels.float() 

            if labels.max() > 1.0:
                labels = labels / 255.0

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    
            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()
            print(f'Epoch {epoch + 1}/{num_epochs}, Step Loss: {loss.item()}')

        #Epoch is over, time to validate
        avg_train_loss = running_loss / len(dataloader)
        avg_val_loss, avg_val_dice = validate(model, validationloader, criterion, device)
        scheduler.step(avg_val_dice)
        print(avg_val_dice)
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model. + {best_val_dice}")

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')