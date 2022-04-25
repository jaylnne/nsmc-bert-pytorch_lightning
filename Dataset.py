from transformers import BertTokenizer
from torch.utils.data import random_split, DataLoader


# TODO NSMCDataset 클래스 추가
    
    
class NSMCDataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir, stem_analyzer, valid_size, max_seq_len, batch_size):
        self.full_data_path = f'{data_dir}/train_{self.stem_analyzer}.csv'
        self.test_data_path = f'{data_dir}/test_{self.stem_analyzer}.csv'
        self.valid_size = valid_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
    def setup(self, stage):
        full = NSMCDataset(self.full_data_path, self.max_seq_len)
        train_size = int(len(full) * (1 - self.valid_size))
        valid_size = len(full) - train_size
        self.train, self.valid = random_split(nsmc_full, [train_size, valid_size])
        self.test = NSMCDataset(self.test_data_path, self.max_seq_len)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=5, shuffle=True, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=64, num_workers=5, shuffle=False, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=64, num_workers=5, shuffle=False, pin_memory=True)
    
    ## TODO predict_dataloader 추가