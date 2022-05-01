import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *


seed_everything(42)

EPOCH = 10
AVAIL_GPUS = min(1, torch.cuda.device_count())

dm = NSMCDataModule(
    data_dir='./data', 
    stem_analyzer='mecab', 
    valid_size=0.1, 
    max_seq_len=64, 
    batch_size=32,
)
dm.setup('fit')
model = NSMCClassification()

trainer = Trainer(
    max_epochs=EPOCH, 
    gpus=AVAIL_GPUS, 
    strategy='ddp',
    callbacks=[EarlyStopping(monitor='val_acc', mode='max')]
)
trainer.fit(model, dm)