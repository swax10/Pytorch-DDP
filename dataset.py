# dataset.py is assumed to contain a class 'CustomDataset' with __len__ and __getitem__ methods

# json for reading json file
import json

# torch for creating and tensors and data loaders
import torch

# Dataset class from torch.utils.data module to create a custom dataset class for the given task
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx]['features'], dtype=torch.float32)
        label = torch.tensor(self.data[idx]['label'], dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
        
        return features, label