import torch
import cv2
import numpy as np
import glob as glob
import os
from tqdm.auto import tqdm
import json

from model import build_model
from datasets import get_datasets, get_data_loaders
from utils import accuracy, precision, recall

# Constants.
DATA_PATH =  'images_yolo' 
LABELS_PATH = 'labels_yolo'
IMAGE_SIZE = 224
DEVICE = 'cpu'


# Load the training and validation datasets.
dataset_train, dataset_valid = get_datasets(True, DATA_PATH, LABELS_PATH) # SET THIS
print(f"[INFO]: Number of training images: {len(dataset_train)}")
print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
# Load the training and validation data loaders.
train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)

# Load the trained model.
model = build_model()
checkpoint = torch.load('model_pretrained_True.pth', map_location=DEVICE, weights_only=False) # FALSE OR TRUE
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])
criterion = checkpoint['loss']
# valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader, criterion)

targets, predictions = [], []
model.eval()
#with torch.no_grad():
   # for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        #image, labels = data
        #targets.append(labels)
        #image = image.to(device)
        #labels = labels.to(device)
        #outputs = model(image)
        #_, preds = torch.max(outputs.data, 1)
        #predictions.append(preds)
        
#metrics = metrics_calculation(targets, predictions)
#metrics.print_metrics()
#metrics.write_json()

counter = 0
with torch.no_grad():
    total_accuracy = 0
    total_recall = 0
    total_precision = 0
    
    for data in valid_loader:
        images, labels = data
        outputs = model(images)
        
        total_accuracy += accuracy(outputs, labels)
        total_recall += recall(outputs, labels)
        total_precision += precision(outputs, labels)
        counter += 1


print(f'Accuracy: {total_accuracy / counter }')
print(f'Recall: {total_recall / counter}')
print(f'Precision: {total_precision / counter}')

data = {}
data["Accuracy"] = total_accuracy / counter
data["Precision"] = total_precision / counter
data["Recall"] = total_recall / counter
with open('metrics.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

