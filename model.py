import torch
## TODO randomness 제어하기 -> torch, CuDNN etc
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import BertModel#, BertTokenizer
from torch.utils.data import random_split, DataLoader


class NSMCClassification(pl.LightningModule):
    
    def __init__(self):
        super(NSMCClassification, self).__init__()
        
        # load pretrained koBERT
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1', output_attentions=True)
        
        # simple linear layer (긍/부정, 2 classes)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)
        self.num_classes = 2
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        
        h, _, attn = self.bert(input_ids=input_id,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                              )
        h_cls = h[:, 0]
        logits = self.W(h_cls)
        
        return logits, attn
    
    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch
        
        # forward
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(y_hat, label)
        
        # logs
        tensorboard_logs = {'train_loss': loss}
        
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch
        
        # forward
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        loss = F.binary_cross_entropy_with_logits(y_hat, label)
        
        # accuracy
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)
        
        return {'val_loss': loss, 'val_acc': val_acc}
    
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {'val_loss': avg_loss,'avg_val_acc':avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}
    
    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch
        
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)
        
        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        test_acc = torch.tensor(test_acc)
        
        return {'test_acc': test_acc}
    
    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        
        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': tensorboard_logs}
    
    def configure_optimizer(self):
        parameters = []
        for p in self.parameters():
            if p.requires_grad:
                parameters.append(p)
            else:
                print(p)
                
        optimizer = torch.optim.Adam(parameters, lr=2e-05, eps=1e-08)
        
        return optimizer
    

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