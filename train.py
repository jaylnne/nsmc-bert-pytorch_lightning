import torch
from datetime import date
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *


seed_everything(42, workers=True)

EPOCH = 10
AVAIL_GPUS = -1
STEM_ANALYZER = 'mecab'
CKPT_PATH = 'checkpoints'

dm = NSMCDataModule(
    data_dir='./data', 
    stem_analyzer=STEM_ANALYZER, 
    valid_size=0.1, 
    max_seq_len=64, 
    batch_size=32,
)
dm.setup('fit')

model = NSMCClassification()

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=CKPT_PATH,
    filename='{epoch:02d}-{val_acc:.3f}',
    verbose=True,
    save_last=False,
    mode='max',
    save_top_k=-1,
)
early_stopping = EarlyStopping(
    monitor='val_acc', 
    mode='max',
)

trainer = Trainer(
    max_epochs=EPOCH,
    accelerator='gpu',
    strategy="ddp",
    devices=AVAIL_GPUS,
    auto_select_gpus=False,
    callbacks=[checkpoint_callback, early_stopping],
    deterministic=True,
)
trainer.fit(model, dm)