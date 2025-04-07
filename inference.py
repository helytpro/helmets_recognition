import torch
import cv2
import numpy as np
import glob as glob
import os
from tqdm.auto import tqdm

from model import build_model
from train import validate
from datasets import get_datasets, get_data_loaders
from utils import save_model, save_plots, save_metrics_plot, metrics_calculation

# Constants.
DATA_PATH =  'images_yolo'  #'../input/test_images'
LABELS_PATH = 'labels_yolo'
IMAGE_SIZE = 224
DEVICE = 'cpu'


# Load the training and validation datasets.
dataset_train, dataset_valid = get_datasets(False, DATA_PATH, LABELS_PATH) # SET THIS
print(f"[INFO]: Number of training images: {len(dataset_train)}")
print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
# Load the training and validation data loaders.
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

# Load the trained model.
model = build_model()
checkpoint = torch.load('model_pretrained_False.pth', map_location=DEVICE, weights_only=False) # FALSE OR TRUE
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
criterion = checkpoint['loss']
# valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion)

targets, predictions = [], []
model.eval()
with torch.no_grad():
    for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        image, labels = data
        targets.append(labels)
        #image = image.to(device)
        #labels = labels.to(device)
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        predictions.append(preds)
        
metrics = metrics_calculation(targets, predictions)
metrics.print_metrics()
metrics.write_json()

