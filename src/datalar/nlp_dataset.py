import torch
from torch.utils.data import  Dataset

#We create a custom dataset class to handle our text data.
class SymptomDataset(Dataset):
    def __init__(self,texts,labels,tokenizer):
        self.texts = texts #The list of symptom descriptions.
        self.labels = labels #The list of corresponding disease labels.
        self.tokenizer = tokenizer #The tokenizer that converts text into numbers.


    def __len__(self): #Returns the total number of samples in the dataset
        return len(self.texts)

    def __getitem__(self, idx):
        
        text = self.texts[idx]
        label = self.labels[idx]


        encoding = self.tokenizer(
            text,
            truncation = True,
            max_length=128
            )
        #We add the label to the encoding.
        encoding['labels'] = torch.tensor(label,dtype=torch.long)

        #We return the encoding.
        return encoding