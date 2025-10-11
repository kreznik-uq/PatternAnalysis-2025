import torch.optim as optim
import torch
import torch.nn as nn
from modules import UNet3D
from torch.utils.data import DataLoader
from dataset import ProstateDataset


if __name__ == '__main__':

    dataset = ProstateDataset(image_dir="semantic_labels_anon", label_dir="semantic_MRs_anon", downsample_factor=0.5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Code reference: https://www.codegenes.net/blog/3d-unet-pytorch/
    model = UNet3D(in_channels=1, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}')