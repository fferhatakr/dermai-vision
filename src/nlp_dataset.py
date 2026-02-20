import torch
from torch.utils.data import  Dataset

class SymptomDataset(Dataset):
    def __init__(self,texts,labels,tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        
        text = self.texts[idx]
        label = self.labels[idx]


        encoding = self.tokenizer(text,
                                  truncation = True,
                                  max_length=128)

        encoding['labels'] = torch.tensor(label,dtype=torch.long)

        return encoding