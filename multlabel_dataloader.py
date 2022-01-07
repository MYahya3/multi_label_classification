import pandas as pd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class MultiLabelDataset(Dataset):
    def __init__(self, csv_dir, data_dir, transform = None):
        self.csv = pd.read_csv(csv_dir)
        self.data_dir = data_dir
        self.images_list = self.csv[:]["Images"]
        self.labels = np.array(self.csv.drop(['Images'], axis=1))
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.csv.iloc[index,0])
        image = Image.open(image_path)
        label = (self.labels[index])
        label = torch.tensor(label)
        image = self.transform(image)

        return (image.float(), label.float())

# csv_dir = "E:/GitHub/pytorch_projects/multi-label_classification/data/Labels_file.csv"
# data_dir = "E:/GitHub/pytorch_projects/multi-label_classification/data/Images"
# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# a = MultiLabelDataset(csv_dir, data_dir, transform=transform)
# print(a[1])