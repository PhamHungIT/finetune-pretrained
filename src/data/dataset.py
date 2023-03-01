
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, text_col, label_col):

        self.labels = df[label_col].tolist()
        self.texts = df[text_col].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        text = self.texts[id]
        label = self.labels[id]
        return text, label
