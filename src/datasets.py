import torch
import os
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Required constants.
# ROOT_DIR = '../input/Chessman-image-dataset/Chess'
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 32
NUM_WORKERS = 4 # Number of parallel processes for data preparation.


# Функция для загрузки данных
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_folder, image_files, label_files, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.image_files = image_files
        self.label_files = label_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        label_path = os.path.join(self.label_folder, self.label_files[idx])
        image = Image.open(img_path).convert('RGB')
        with open(label_path, 'r') as f:
            label = int(f.read().strip())  # Предполагаем, что метка хранится в текстовом файле
        if self.transform:
            image = self.transform(image)
        return image, label


# Training transforms
def get_train_transform(IMAGE_SIZE, pretrained):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return train_transform

# Validation transforms
def get_valid_transform(IMAGE_SIZE, pretrained):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize_transform(pretrained)
    ])
    return valid_transform

# Image normalization transforms.
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weights.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    else: # Normalization when training from scratch.
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    return normalize

def get_datasets(imagePath="images", labelPath="labels", pretrained=True):
    image_files = [f for f in os.listdir(imagePath) if f.endswith(('jpg', 'png', 'jpeg'))]
    label_files = [f for f in os.listdir(labelPath) if f.endswith('txt')]

    assert len(image_files) == len(label_files), "Количество изображений и меток должно совпадать."

    train_images, test_images, train_labels, test_labels = train_test_split(
    image_files, label_files, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(imagePath, labelPath,
                                train_images, train_labels,
                                transform=get_train_transform(IMAGE_SIZE, pretrained))
    test_dataset = CustomDataset(imagePath, 
                                labelPath, test_images, test_labels,
                                transform=get_valid_transform(IMAGE_SIZE, pretrained))

    return train_dataset, test_dataset

def get_data_loaders(dataset_train, dataset_valid):
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, 
        shuffle=True)
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, 
        shuffle=False)
    return train_loader, valid_loader 