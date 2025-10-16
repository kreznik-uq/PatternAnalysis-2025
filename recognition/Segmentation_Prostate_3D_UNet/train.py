import torch.optim as optim
import torch
import torch.nn as nn
from modules import UNet3D
from torch.utils.data import DataLoader
from dataset import ProstateDataset
from torch.amp import GradScaler, autocast


if __name__ == '__main__':

    dataset = ProstateDataset(image_dir="processed_data/images", label_dir="processed_data/labels")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Code reference: https://www.codegenes.net/blog/3d-unet-pytorch/
    model = UNet3D(in_channels=1, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    num_epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler() 

    print("started")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
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

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')