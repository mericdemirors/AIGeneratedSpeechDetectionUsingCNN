# Creates Feature Extracted dataset using resnet50 model from Pytroch.

print("imports...", flush=True)
import os
import random
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models

import warnings
warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split

# Create the folder to hold image datatset
root_dir = ""
os.makedirs(os.path.join(root_dir, "extracted_featrues"))
for feature_folder in ["absolutes","angles","cum3s","imags","reals"]:
    for folder in ["fake_audio", "real_audio"]:
        os.makedirs(os.path.join(root_dir, "extracted_featrues", feature_folder, folder))

# get the resnet50 pre-trained model to pass images to get the extracted features
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class PretrainedResNet(nn.Module):
    def __init__(self):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
    def forward(self, x):
        return self.model(x)
fe_model = PretrainedResNet().eval().to(device)

# extract features of all image dataset and save them as .npy
for feature_folder in ["absolutes","angles","cum3s","imags","reals"]:
    fake_path = os.path.join(root_dir, feature_folder, "fake_audio")
    real_path = os.path.join(root_dir, feature_folder, "real_audio")

    fake_images = [os.path.join(fake_path, x) for x in os.listdir(fake_path)]
    real_images = [os.path.join(real_path, x) for x in os.listdir(real_path)]

    X = fake_images + real_images
    Y = [1 for _ in fake_images] + [0 for _ in real_images]

    for x,y in tqdm(zip(X,Y), desc="Extracting features from " + feature_folder):
        image = cv2.imread(x).astype(np.float32) / 255
        image = torch.from_numpy(np.moveaxis(image, 2, 0)[np.newaxis, ...]).to(device)

        with torch.no_grad():
            features = fe_model(image)[0].cpu().numpy()

        save_path = x.replace(os.path.join(root_dir, feature_folder), os.path.join(root_dir, "extracted_featrues", feature_folder))[:-4]
        np.save(save_path, features)