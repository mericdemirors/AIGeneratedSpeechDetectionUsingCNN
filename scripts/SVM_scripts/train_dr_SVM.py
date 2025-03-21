# Trains dimension reduction SVMs on a dataset, and saves the evaluation scores in a txt

print("imports...", flush=True)
import os
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
    parser.add_argument('-dim_red', '-dr', type=str, default="", help='Method to use at dimension reduction. default is: ""')

    return parser.parse_args()

args = parse_arguments()
model_type = args.model
root_dir = args.path
method = args.dim_red

model_serial_number = "dim_red_" + model_type + "_" + os.path.split(root_dir)[1] + "_" + method + "_" + datetime.now().strftime("%m_%d_%H_%M_%S")
print("serial_number:", model_serial_number, "\ndataset:", root_dir, flush=True)
model_serial_path = os.path.join("", model_serial_number)
os.makedirs(model_serial_path)


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
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from umap import UMAP
def fit_PCA(images, num_of_features):
    pca = PCA(n_components=num_of_features, random_state=42)
    pca.fit(images)
    return pca

def fit_ICA(images, num_of_features):
    ica = FastICA(n_components=num_of_features)
    ica.fit(images)
    return ica

def fit_UMAP(images, num_of_features):
    umap = UMAP(n_components=num_of_features)
    umap.fit(images)
    return umap

def dim_red(method, images, num_of_features=256):
    if method == "PCA":
        method = fit_PCA(images, num_of_features)
        return method, images
    elif method == "ICA":
        method = fit_ICA(images, num_of_features)
        return method, images
    elif method == "UMAP":
        method = fit_UMAP(images, num_of_features)
        return method, images
    
    features = method.transform(images)
    return method, features

def load_batch(Xs, Ys):
    images = [(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.375, fy=0.375).astype(np.float16) / 255).flatten() for path in tqdm(Xs, desc="loading images")]
    return images, Ys

train_images, trian_labels = load_batch(x_train, y_train)
method, train_images = dim_red(method, train_images) # initializing images
method, train_images = dim_red(method, train_images)

model.fit(train_images, trian_labels)

validation_images, validation_labels = load_batch(x_validation, y_validation)
method, validation_images = dim_red(method, validation_images)

predictions = model.predict(validation_images)
predictions = np.array(predictions)
real_values = np.array(validation_labels)

class_correct, class_total = [0,0], [0,0]
correct = (predictions == real_values).squeeze()
for e,label in enumerate(real_values):
    class_correct[label] += correct[e].item()
    class_total[label] += 1
validation_acc = sum(class_correct)/sum(class_total)

print("Total accuracy:", sum(class_correct)/sum(class_total))
print("Real detection: ", class_correct[0], "/", class_total[0], " | accuracy:", class_correct[0]/class_total[0], sep="")
print("Fake detection: ", class_correct[1], "/", class_total[1], " | accuracy:", class_correct[1]/class_total[1], sep="")

joblib.dump(model, os.path.join(model_serial_path, model_type+".joblib"))

### ---|---|---|---|---|---|---|---|---|---|--- TESTING ---|---|---|---|---|---|---|---|---|---|--- ###
print("testing...", flush=True)
loaded_model = joblib.load(os.path.join(model_serial_path, model_type+".joblib"))
images, labels = load_batch(x_test, y_test)
method, images = dim_red(method, images)

predictions = model.predict(images)

predictions = np.array(predictions)
real_values = np.array(labels)

class_correct, class_total = [0,0], [0,0]
correct = (predictions == real_values).squeeze()
for e,label in enumerate(real_values):
    class_correct[label] += correct[e].item()
    class_total[label] += 1

print("Total accuracy:", sum(class_correct)/sum(class_total))
print("Real detection: ", class_correct[0], "/", class_total[0], " | accuracy:", class_correct[0]/class_total[0], sep="")
print("Fake detection: ", class_correct[1], "/", class_total[1], " | accuracy:", class_correct[1]/class_total[1], sep="")

with open(os.path.join(model_serial_path, "accuracy.txt"), 'w') as txt:
    txt.write("Total accuracy: " + str(sum(class_correct)/sum(class_total)) + "\n")
    txt.write("Real detection: " + str(class_correct[0]) + "/" + str(class_total[0]) + " | accuracy: " + str(class_correct[0]/class_total[0]) + "\n")
    txt.write("Fake detection: " + str(class_correct[1]) + "/" + str(class_total[1]) + " | accuracy: " + str(class_correct[1]/class_total[1]) + "\n")
