import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Custom dataset
class MyTokenizer(Dataset):
    def __init__(self, data, path_to_hub):
        self.tokenizer = AutoTokenizer.from_pretrained(path_to_hub) # Load your tokenizer here with a pre-specified vocabulary
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data = data # a dataframe
        self.content_name = 'instruction'
        self.target_name = 'code'
        self.list_test = 'test_list'
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        text = self.data.iloc[index][self.content_name]
        code = self.data.iloc[index][self.target_name]
        test_list = self.data.iloc[index][self.list_test]

        inputs = self.tokenizer(text, padding=True)
        label = self.tokenizer.encode(code, padding=True)
        tests = self.tokenizer.encode(test_list, padding=True)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long),
            'tests' : torch.tensor(tests, dtype=torch.long)
        }
    

def dataloading(data, path_to_hub, batch_size, num_workers, g, seed_worker):
    # create dataset out of dataframe
    dataset = MyTokenizer(data, path_to_hub)

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        )
    
    return dataloader

