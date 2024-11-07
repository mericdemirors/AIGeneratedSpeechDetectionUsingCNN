# Checks if there are any corrupted image files in the dataset.
# Normaly there is no need to run this code at any moment in a safe environment,
# but since the system, that research is done with, is not that much safe to trust with datasets,
# we ran this few times to check if everyting is safe

import os
import cv2
import numpy as np
from tqdm import tqdm

path_to_check = ""
feature_folders = (os.path.join(path_to_check, x) for x in os.listdir(path_to_check))
class_folders = ["fake_audio", "real_audio"]

corrupted = []
print("read all images and check their shapes and if there is any Nans or infs")
for ff in feature_folders: # check each folder of 5 feature
    for cf in class_folders: # check real and fake folders
        image_folder_path = os.path.join(ff,cf)
        print("checking", image_folder_path)
        image_paths = [os.path.join(image_folder_path, i) for i in os.listdir(image_folder_path)]
        for ip in tqdm(image_paths):
            img = cv2.imread(ip)
            # it is 401 because of the stingray library output format
            if ((img.shape != (401,401,3)) or (np.isnan(img).sum() != 0) or (np.isinf(img).sum() != 0)):
                corrupted.append(ip)

print("total of", len(corrupted), "corrupted files")
with open(r'corrupteds.txt', 'w') as fp:
    for corrupted in corrupted:
        fp.write(corrupted + "\n")
