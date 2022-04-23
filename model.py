from transformers import BertModel
import pytorch_lightning as pl
import torch.nn.functional as F


class NSMCClassification(pl.LightningModule):
    
    def __init__(self):
        super(NSMC_Finetuner, self).__init__()
        
        # load pretrained koBERT
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1', output_attentions=True)
        
        # simple linear layer (긍/부정, 2 classes)
        self.W = nn.Linear(bert.config.hidden_size, 2)
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
        
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc':avg_val_acc}
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