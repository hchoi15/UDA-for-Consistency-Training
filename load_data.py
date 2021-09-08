import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class IMDBDataset(Dataset):
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        return x


dataset = IMDBDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)