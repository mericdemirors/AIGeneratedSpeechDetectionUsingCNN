# Tests a model checkpoint on a dataset, and saves the evaluation scores in a txt

print("imports...", flush=True)
import os
import time
import argparse

import torch
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore")

from datasets import *
from models import *

import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, random_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-model', '-m', type=str, default="basic_CNN", help='model type to train. default is: \"basic_CNN\"')
    parser.add_argument('-path', '-p', type=str, default="", help='Root directory of images. default is: \"\"')
    parser.add_argument('-dataset', '-d', type=str, default="gray", help='Dataset type to train on. default is: \"gray\"')
    parser.add_argument('-split', '-s', type=float, nargs=2, default=(0.8,0.1), help='Train validation split as two floats separated by space(remaining will be test). default is: (0.8,0.1)')
    parser.add_argument('-epochs', '-e', type=int, default=100, help='Number of epochs. default is: 100')
    parser.add_argument('-batch', '-b', type=int, default=32, help='Batch size. default is: 32')
    parser.add_argument('-LR', '-lr', type=float, default=0.001, help='Learning rate. default is: 0.001')
    parser.add_argument('-threshold', '-th', type=float, default=0.5, help='Threshold for sigmoid classification. default is: 0.5')
    parser.add_argument('-checkpoint', '-c', type=str, default="", help='checkpoint to load. default is ""')
    return parser.parse_args()


args = parse_arguments()
model_type = args.model
root_dir = args.path
dataset_type = args.dataset
TRAIN_VALIDATION_SPLIT = tuple(args.split)
NUM_OF_EPOCHS = args.epochs
BATCH_SIZE = args.batch
LR = args.LR
THRESHOLD = args.threshold
ckpt = args.checkpoint

model_serial_path = os.path.split(ckpt)[0]
### ---|---|---|---|---|---|---|---|---|---|--- DATASET & MODEL ---|---|---|---|---|---|---|---|---|---|--- ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = import_dataset(dataset_type, root_dir)

train_size = int(TRAIN_VALIDATION_SPLIT[0] * len(dataset))
validation_size = int(TRAIN_VALIDATION_SPLIT[1] * len(dataset))
test_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

if "transform" in dataset_type:
    all_images = np.array([x for x,y in tqdm(train_dataset, desc="Dimension reduction")])
    train_dataset.dataset.method, _ = dim_red(train_dataset.dataset.method, all_images)
    validation_dataset.dataset.method = train_dataset.dataset.method
    test_dataset.dataset.method = train_dataset.dataset.method

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


model = import_model(model_type).to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

### ---|---|---|---|---|---|---|---|---|---|--- TESTING ---|---|---|---|---|---|---|---|---|---|--- ###
print("testing...", flush=True)
loaded_model = import_model(model_type = args.model).to(device)
loaded_model.load_state_dict(torch.load(ckpt))
loaded_model.eval()

start_time = time.time()
with torch.no_grad():
    for test_data, test_labels in test_dataloader:
        test_data, test_labels = test_data.to(device), test_labels.to(device)

        test_outputs = loaded_model(test_data)
        
end_time = time.time()
elapsed_time = end_time - start_time

with open(os.path.join(model_serial_path, "inference.txt"), 'w') as txt:
    txt.write("Elapsed time: " + str(elapsed_time) + " seconds" + "\n")

torch.cuda.empty_cache()

