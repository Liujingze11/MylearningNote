from torch.utils.data import Dataset
import torch
import pandas as pd

def load_data(args, split):
    df = pd.read_csv(f"{args.data_root}/{split}.csv")
    texts = df['text'].values.tolist()
    labels = df['target'].values.tolist()
    return texts, labels

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, is_test):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = data[0]
        self.labels = data[1]
        self.is_test = is_test
            
    def __len__(self):
        """returns the length of dataframe"""
        return len(self.texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        text = str(self.texts[index])
        source = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        data_sample = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
        }
        if not self.is_test:
            label = self.labels[index]
            target_ids = torch.tensor(label).squeeze()
            data_sample["labels"] = target_ids 
        return data_sample
            