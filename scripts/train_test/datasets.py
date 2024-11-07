# Here are all the datasets used in the research, and needed functions for them

import os
import cv2
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset

class AudioDataset_gray(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fake_path = os.path.join(root_dir, "fake_audio")
        self.real_path = os.path.join(root_dir, "real_audio")

        self.fake_images = os.listdir(self.fake_path)
        self.real_images = os.listdir(self.real_path)

        self.x = self.fake_images + self.real_images
        self.y = [1 for _ in self.fake_images] + [0 for _ in self.real_images]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_name = self.x[idx]
        image_class = self.y[idx]

        if image_class == 1: # fake_audio
            image_path = os.path.join(self.root_dir, "fake_audio", image_name)
        elif image_class == 0: # real_audio
            image_path = os.path.join(self.root_dir, "real_audio", image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)[np.newaxis, ...] / 255

        return image, image_class

class AudioDataset_RGB(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fake_path = os.path.join(root_dir, "fake_audio")
        self.real_path = os.path.join(root_dir, "real_audio")

        self.fake_images = os.listdir(self.fake_path)
        self.real_images = os.listdir(self.real_path)

        self.x = self.fake_images + self.real_images
        self.y = [1 for _ in self.fake_images] + [0 for _ in self.real_images]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_name = self.x[idx]
        image_class = self.y[idx]

        if image_class == 1: # fake_audio
            image_path = os.path.join(self.root_dir, "fake_audio", image_name)
        elif image_class == 0: # real_audio
            image_path = os.path.join(self.root_dir, "real_audio", image_name)

        image = cv2.imread(image_path).astype(np.float32) / 255
        image = np.moveaxis(image, 2, 0)

        return image, image_class

class AudioDataset_multi(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        feature_folders = [os.path.join(root_dir,x) for x in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir,x))]

        # 5 features image paths are held together
        real_folder_file_dict, fake_folder_file_dict = {}, {}
        for ff in feature_folders:
            real_folder_file_dict[ff] = sorted([os.path.join(ff,"real_audio",x) for x in os.listdir(os.path.join(ff, "real_audio"))])
            fake_folder_file_dict[ff] = sorted([os.path.join(ff,"fake_audio",x) for x in os.listdir(os.path.join(ff, "fake_audio"))])

        self.real_multi_feature_paths = list(zip(*real_folder_file_dict.values()))
        self.fake_multi_feature_paths = list(zip(*fake_folder_file_dict.values()))


        self.x = self.fake_multi_feature_paths + self.real_multi_feature_paths
        self.y = [1 for _ in self.fake_multi_feature_paths] + [0 for _ in self.real_multi_feature_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_paths = self.x[idx]
        image_class = self.y[idx]

        # 5 features image paths are read together
        image = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255 for path in feature_paths])

        return image, image_class

class AudioDataset_gray_flat(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fake_path = os.path.join(root_dir, "fake_audio")
        self.real_path = os.path.join(root_dir, "real_audio")

        self.fake_images = os.listdir(self.fake_path)
        self.real_images = os.listdir(self.real_path)

        self.x = self.fake_images + self.real_images
        self.y = [1 for _ in self.fake_images] + [0 for _ in self.real_images]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_name = self.x[idx]
        image_class = self.y[idx]

        if image_class == 1: # fake_audio
            image_path = os.path.join(self.root_dir, "fake_audio", image_name)
        elif image_class == 0: # real_audio
            image_path = os.path.join(self.root_dir, "real_audio", image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)[np.newaxis, ...] / 255

        return image.flatten(), image_class

class AudioDataset_multi_flat(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        feature_folders = [os.path.join(root_dir,x) for x in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir,x))]

        # 5 features image paths are held together
        real_folder_file_dict, fake_folder_file_dict = {}, {}
        for ff in feature_folders:
            real_folder_file_dict[ff] = sorted([os.path.join(ff,"real_audio",x) for x in os.listdir(os.path.join(ff, "real_audio"))])
            fake_folder_file_dict[ff] = sorted([os.path.join(ff,"fake_audio",x) for x in os.listdir(os.path.join(ff, "fake_audio"))])

        self.real_multi_feature_paths = list(zip(*real_folder_file_dict.values()))
        self.fake_multi_feature_paths = list(zip(*fake_folder_file_dict.values()))


        self.x = self.fake_multi_feature_paths + self.real_multi_feature_paths
        self.y = [1 for _ in self.fake_multi_feature_paths] + [0 for _ in self.real_multi_feature_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_paths = self.x[idx]
        image_class = self.y[idx]

        # 5 features image paths are read together, resize is done to fit the data into RAM
        image = np.array([(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.25, fy=0.25).astype(np.float32)/255).flatten() for path in feature_paths]).flatten()

        return image, image_class


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
        return method, None
    elif method == "ICA":
        method = fit_ICA(images, num_of_features)
        return method, None
    elif method == "UMAP":
        method = fit_UMAP(images, num_of_features)
        return method, None
    
    features = method.transform(images)
    return method, features

class AudioDataset_transform(Dataset):
    def __init__(self, root_dir, method):
        self.root_dir = root_dir
        self.method = method
        self.fake_path = os.path.join(root_dir, "fake_audio")
        self.real_path = os.path.join(root_dir, "real_audio")

        self.fake_images = os.listdir(self.fake_path)
        self.real_images = os.listdir(self.real_path)
        
        self.x = self.fake_images + self.real_images
        self.y = [1 for _ in self.fake_images] + [0 for _ in self.real_images]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_name = self.x[idx]
        image_class = self.y[idx]

        if image_class == 1: # fake_audio
            image_path = os.path.join(self.root_dir, "fake_audio", image_name)
        elif image_class == 0: # real_audio
            image_path = os.path.join(self.root_dir, "real_audio", image_name)

        # resizing is used to fit the data to RAM
        image = (cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.125, fy=0.125).astype(np.float16)/255).flatten()
        
        if type(self.method) == str: # if dim red method is not initialized with training data return images for init
            return image, image_class
        else: # else transform images
            self.method, features = dim_red(self.method, image.reshape(1, -1))
            features = np.array(features).astype(np.float32).reshape(256)
            return features, image_class

class AudioDataset_transform_multi(Dataset):
    def __init__(self, root_dir, method):
        self.root_dir = root_dir
        self.method = method

        # 5 features' image paths are held together
        feature_folders = [os.path.join(root_dir,x) for x in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir,x))]

        real_folder_file_dict, fake_folder_file_dict = {}, {}
        for ff in feature_folders:
            real_folder_file_dict[ff] = sorted([os.path.join(ff,"real_audio",x) for x in os.listdir(os.path.join(ff, "real_audio"))])
            fake_folder_file_dict[ff] = sorted([os.path.join(ff,"fake_audio",x) for x in os.listdir(os.path.join(ff, "fake_audio"))])

        self.real_multi_feature_paths = list(zip(*real_folder_file_dict.values()))
        self.fake_multi_feature_paths = list(zip(*fake_folder_file_dict.values()))

        self.x = self.fake_multi_feature_paths + self.real_multi_feature_paths
        self.y = [1 for _ in self.fake_multi_feature_paths] + [0 for _ in self.real_multi_feature_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_paths = self.x[idx]
        image_class = self.y[idx]

        # 5 features' image paths are read together, resizing is used to fit the data to RAM
        images = np.array([(cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (0,0), fx=0.25, fy=0.25).astype(np.float16)/255).flatten() for path in feature_paths]).flatten()
        
        if type(self.method) == str: # if dim red method is not initialized with training data return images for init
            return images, image_class
        else: # else transform images
            self.method, features = dim_red(self.method, images.reshape(1, -1))
            features = np.array(features).astype(np.float32).reshape(256)
            return features, image_class

class AudioDataset_FE(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fake_path = os.path.join(root_dir, "fake_audio")
        self.real_path = os.path.join(root_dir, "real_audio")

        self.fake_images = os.listdir(self.fake_path)
        self.real_images = os.listdir(self.real_path)

        self.x = self.fake_images + self.real_images
        self.y = [1 for _ in self.fake_images] + [0 for _ in self.real_images]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image_name = self.x[idx]
        image_class = self.y[idx]

        if image_class == 1: # fake_audio
            image_path = os.path.join(self.root_dir, "fake_audio", image_name)
        elif image_class == 0: # real_audio
            image_path = os.path.join(self.root_dir, "real_audio", image_name)

        image = np.load(image_path)

        return image, image_class

class AudioDataset_FE_multi(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        feature_folders = [os.path.join(root_dir,x) for x in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir,x))]

        # 5 features' image paths are held together
        real_folder_file_dict, fake_folder_file_dict = {}, {}
        for ff in feature_folders:
            real_folder_file_dict[ff] = sorted([os.path.join(ff,"real_audio",x) for x in os.listdir(os.path.join(ff, "real_audio"))])
            fake_folder_file_dict[ff] = sorted([os.path.join(ff,"fake_audio",x) for x in os.listdir(os.path.join(ff, "fake_audio"))])

        self.real_multi_feature_paths = list(zip(*real_folder_file_dict.values()))
        self.fake_multi_feature_paths = list(zip(*fake_folder_file_dict.values()))


        self.x = self.fake_multi_feature_paths + self.real_multi_feature_paths
        self.y = [1 for _ in self.fake_multi_feature_paths] + [0 for _ in self.real_multi_feature_paths]
        self.num_samples = len(self.y)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feature_paths = self.x[idx]
        image_class = self.y[idx]

        # 5 features' image paths are read together
        image = np.array([np.load(path).flatten() for path in feature_paths]).flatten()

        return image, image_class



def import_dataset(dataset_type, root_dir, method=""):
    if dataset_type == "gray":
        return AudioDataset_gray(root_dir)
    elif dataset_type == "RGB":
        return AudioDataset_RGB(root_dir)
    elif dataset_type == "multi":
        return AudioDataset_multi(root_dir)
    elif dataset_type == "gray_flat":
        return AudioDataset_gray_flat(root_dir)
    elif dataset_type == "multi_flat":
        return AudioDataset_multi_flat(root_dir)
    elif "transform" in dataset_type and "multi" not in dataset_type:
        method = dataset_type.split(" ")[1]
        return AudioDataset_transform(root_dir, method)
    elif "transform" in dataset_type and "multi" in dataset_type:
        method = dataset_type.split(" ")[1]
        return AudioDataset_transform_multi(root_dir, method)
    elif dataset_type == "FE":
        return AudioDataset_FE(root_dir)
    elif dataset_type == "multi_FE":
        return AudioDataset_FE_multi(root_dir)
