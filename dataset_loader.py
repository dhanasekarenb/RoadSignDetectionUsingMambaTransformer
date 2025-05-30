import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch

class TrafficSignDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Traffic Sign Dataset Loader.
        
        :param csv_file: Path to the CSV file containing image paths and labels.
        :param img_dir: Root directory where images are stored.
        :param transform: Image transformations.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx]['Path'])

        # Check if image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load Image
        image = Image.open(img_path).convert("RGB")
        label = int(self.data.iloc[idx]['ClassId'])

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# ✅ Data transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to fixed size
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

def get_dataloaders(batch_size=64, train_split=0.8):
    """
    Create train and validation DataLoaders.

    :param batch_size: Batch size for training & validation.
    :param train_split: Percentage of dataset to be used for training.
    :return: train_loader, val_loader
    """
    dataset = TrafficSignDataset("dataset/Train.csv", "dataset", transform=transform)
    
    # Split dataset into train and validation
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

# Test the dataset loading
if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Display sample batch
    images, labels = next(iter(train_loader))
    print(f"Sample batch shape: {images.shape}, Labels: {labels[:5]}")
