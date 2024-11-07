# Here are all the models used in the research

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class PretrainedGoogleNet(nn.Module):
    def __init__(self):
        super(PretrainedGoogleNet, self).__init__()
        self.model = models.googlenet(pretrained=True)

        # freezing parameters before inception4a
        for i,p in enumerate(self.model.parameters()):
            if i < 45:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)

        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256), nn.ReLU(), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

class PretrainedResNet(nn.Module):
    def __init__(self):
        super(PretrainedResNet, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        # freezing parameters before (layer2)
        for i,p in enumerate(self.model.parameters()):
            if i < 33:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
                
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())
    
    def forward(self, x):
        return self.model(x)

class GoogleNetGray(nn.Module):
    def __init__(self):
        super(GoogleNetGray, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
        self.model.conv1.conv = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class ResNetGray(nn.Module):
    def __init__(self):
        super(ResNetGray, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class GoogleNetRGB(nn.Module):
    def __init__(self):
        super(GoogleNetRGB, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class ResNetRGB(nn.Module):
    def __init__(self):
        super(ResNetRGB, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class GoogleNetMulti(nn.Module):
    def __init__(self):
        super(GoogleNetMulti, self).__init__()
        self.model = models.googlenet(pretrained=False, num_classes=1)
        self.model.conv1.conv = nn.Conv2d(5, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        if self.training:
            return nn.Sigmoid()(self.model(x).logits)
        else:
            return nn.Sigmoid()(self.model(x))

class ResNetMulti(nn.Module):
    def __init__(self):
        super(ResNetMulti, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=1)
        self.model.conv1 = nn.Conv2d(5, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=True)
    
    def forward(self, x):
        return nn.Sigmoid()(self.model(x))

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 16, kernel_size=3)
        self.fully_conv = nn.Conv2d(16, 1, kernel_size=8)

        self.relu = F.relu
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(self.relu(self.bn1(self.conv2(x))))
        x = self.max_pool(self.relu(self.bn2(self.conv3(x))))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.sigmoid(self.fully_conv(x))
        
        return x[:,0,0]
    
class BasicCNNMulti(nn.Module):
    def __init__(self):
        super(BasicCNNMulti, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3)
        self.conv5 = nn.Conv2d(64, 16, kernel_size=3)
        self.fully_conv = nn.Conv2d(16, 1, kernel_size=8)

        self.relu = F.relu
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(self.relu(self.bn1(self.conv2(x))))
        x = self.max_pool(self.relu(self.bn2(self.conv3(x))))
        x = self.max_pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.sigmoid(self.fully_conv(x))
        
        return x[:,0,0]
    
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3)
        self.drop4 = nn.Dropout2d(p=0.2)
        self.conv5 = nn.Conv2d(64, 16, kernel_size=3)
        self.drop5 = nn.Dropout2d(p=0.2)
        self.fully_conv = nn.Conv2d(16, 1, kernel_size=8)

        self.relu = F.relu
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(self.relu(self.bn1(self.conv2(x))))
        x = self.max_pool(self.relu(self.bn2(self.conv3(x))))
        x = self.max_pool(self.drop4(self.relu(self.conv4(x))))
        x = self.drop5(self.relu(self.conv5(x)))
        x = self.sigmoid(self.fully_conv(x))
        
        return x[:,0,0]

class ComplexCNNMulti(nn.Module):
    def __init__(self):
        super(ComplexCNNMulti, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=3)
        self.drop4 = nn.Dropout2d(p=0.2)
        self.conv5 = nn.Conv2d(64, 16, kernel_size=3)
        self.drop5 = nn.Dropout2d(p=0.2)
        self.fully_conv = nn.Conv2d(16, 1, kernel_size=8)

        self.relu = F.relu
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.max_pool(self.relu(self.bn1(self.conv2(x))))
        x = self.max_pool(self.relu(self.bn2(self.conv3(x))))
        x = self.max_pool(self.drop4(self.relu(self.conv4(x))))
        x = self.drop5(self.relu(self.conv5(x)))
        x = self.sigmoid(self.fully_conv(x))
        
        return x[:,0,0]

class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.fc1 = nn.Linear(401*401, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class BasicNNMulti(nn.Module):
    def __init__(self):
        super(BasicNNMulti, self).__init__()
        self.fc1 = nn.Linear(50000, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)

        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class DimRedNN(nn.Module):
    def __init__(self):
        super(DimRedNN, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class DimRedNNMulti(nn.Module):
    def __init__(self):
        super(DimRedNNMulti, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

class FENN(nn.Module):
    def __init__(self):
        super(FENN, self).__init__()
        self.fc1 = nn.Linear(1000, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

class FENNMulti(nn.Module):
    def __init__(self):
        super(FENNMulti, self).__init__()
        self.fc1 = nn.Linear(5000, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

        self.relu = F.relu
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x

def import_model(model_type):
    if model_type == "pt_google":
        return PretrainedGoogleNet()
    elif model_type == "pt_resnet":
        return PretrainedResNet()
    elif model_type == "google_gray":
        return GoogleNetGray()
    elif model_type == "resnet_gray":
        return ResNetGray()
    elif model_type == "google_RGB":
        return GoogleNetRGB()
    elif model_type == "resnet_RGB":
        return ResNetRGB()
    elif model_type == "google_multi":
        return GoogleNetMulti()
    elif model_type == "resnet_multi":
        return ResNetMulti()
    elif model_type == "basic_CNN":
        return BasicCNN()
    elif model_type == "basic_CNN_multi":
        return BasicCNNMulti()
    elif model_type == "basic_NN":
        return BasicNN()
    elif model_type == "basic_NN_multi":
        return BasicNNMulti()
    elif model_type == "dim_red_NN":
        return DimRedNN()
    elif model_type == "dim_red_NN_multi":
        return DimRedNNMulti()
    elif model_type == "feat_ext_NN":
        return FENN()
    elif model_type == "feat_ext_NN_multi":
        return FENNMulti()
    elif model_type == "complex_CNN":
        return ComplexCNN()
    elif model_type == "complex_CNN_multi":
        return ComplexCNNMulti()
