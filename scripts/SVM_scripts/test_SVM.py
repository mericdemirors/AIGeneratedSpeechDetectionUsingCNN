# Tests SVMs on a dataset, and saves the evaluation scores in a txt

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
fake_path = os.path.join(root_dir, "fake_audio")
real_path = os.path.join(root_dir, "real_audio")

fake_images = [os.path.join(fake_path, x) for x in os.listdir(fake_path)]
real_images = [os.path.join(real_path, x) for x in os.listdir(real_path)]

X = fake_images + real_images
Y = [1 for _ in fake_images] + [0 for _ in real_images]

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
    images = [(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.375, fy=0.375).astype(np.float16) / 255).flatten() for path in tqdm(Xs, desc="loading images")]
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