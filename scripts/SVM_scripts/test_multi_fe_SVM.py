# Tests feature extraction multi SVMs on a dataset, and saves the evaluation scores in a txt

print("imports...", flush=True)
import os
import time
import random
import argparse
from tqdm import tqdm
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import cv2
import joblib
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-model', '-m', type=str, default="linear_SVC", help='model type to train. default is: "linear_SVC"')
    parser.add_argument('-path', '-p', type=str, default="", help='Root directory of images. default is: ""')

    return parser.parse_args()

args = parse_arguments()
model_type = args.model
root_dir = args.path

model_serial_path = model_type
### ---|---|---|---|---|---|---|---|---|---|--- DATASET & MODEL ---|---|---|---|---|---|---|---|---|---|--- ###
feature_folders = [os.path.join(root_dir,x) for x in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir,x))]

real_folder_file_dict, fake_folder_file_dict = {}, {}
for ff in feature_folders:
    real_folder_file_dict[ff] = sorted([os.path.join(ff,"real_audio",x) for x in os.listdir(os.path.join(ff, "real_audio"))])
    fake_folder_file_dict[ff] = sorted([os.path.join(ff,"fake_audio",x) for x in os.listdir(os.path.join(ff, "fake_audio"))])

real_multi_feature_paths = list(zip(*real_folder_file_dict.values()))
fake_multi_feature_paths = list(zip(*fake_folder_file_dict.values()))

X = fake_multi_feature_paths + real_multi_feature_paths
Y = [1 for _ in fake_multi_feature_paths] + [0 for _ in real_multi_feature_paths]

combined = list(zip(X, Y))
random.shuffle(combined)
X,Y = zip(*combined)

x_train, x_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
x_validation, x_test, y_validation, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

if model_type == "linear_SVC":
    model = svm.LinearSVC(dual=True, max_iter=1000, verbose=1, random_state=42)
elif model_type == "SVC":
    model = svm.SVC(max_iter=1000, verbose=1, random_state=42)

### ---|---|---|---|---|---|---|---|---|---|--- TRAINING ---|---|---|---|---|---|---|---|---|---|--- ###
def load_batch(Xs, Ys):
    images = [np.array([
             np.load(path)
             for path in feature_paths]).flatten() 
             for feature_paths in tqdm(Xs, desc="loading images")]

    # here is above list comprehension
    # for feature_paths in Xs:
    #     for path in feature_paths:
    #         read images and flatten them
    #         stack 5 of them back to back
    #         flatten the whole thing
    #         now we have a input vector that has 5 flattened images of size(401x401) total input size: 804005

    return images, Ys

### ---|---|---|---|---|---|---|---|---|---|--- TESTING ---|---|---|---|---|---|---|---|---|---|--- ###
print("testing...", flush=True)
loaded_model = joblib.load(os.path.join(model_serial_path, model_type+".joblib"))
images, labels = load_batch(x_test, y_test)

print("testing...", flush=True)
loaded_model = joblib.load(model_type+".joblib")
images, labels = load_batch(x_test, y_test)

start_time = time.time()

predictions = loaded_model.predict(images)

end_time = time.time()
elapsed_time = end_time - start_time

with open(os.path.join(os.path.split(model_type)[0], "inference.txt"), 'w') as txt:
    txt.write("Elapsed time: " + str(elapsed_time) + " seconds" + "\n")