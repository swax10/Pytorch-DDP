# dataset.py is assumed to contain a class 'CustomDataset' with __len__ and __getitem__ methods

# json for reading json file
import json

# torch for creating and tensors and data loaders
import torch

# Dataset class from torch.utils.data module to create a custom dataset class for the given task
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['features'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.float32)
        return features, label