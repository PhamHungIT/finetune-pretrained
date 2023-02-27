
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, label2id):

        self.label_ids = [label2id[label] for label in df['category']]
        self.texts = df['text'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        text = self.texts(id)
        label = self.label_ids(id)
        return text, label
    
from sklearn.feature_extraction.text import CountVectorizer

def text_pipeline(text, vectorizer):
    """
    Input as a text and return list of indexes respectively

    Args:
        text: (str) 
    """
    return vectorizer.transform([text]).toarray()[0]

def label_pipeline(label, label2id):
    return label2id[label] 



def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)
