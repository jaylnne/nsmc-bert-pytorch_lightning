import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *


seed_everything(42, workers=True)

EPOCH = 10
AVAIL_GPUS = -1
STEM_ANALYZER = 'mecab'
CKPT_SAVE_PATH = 'checkpoints'
VALID_SIZE = 0.1
MAX_SEQ_LEN = 64
BATCH_SIZE = 32

dm = NSMCDataModule(
    data_dir='./data', 
    stem_analyzer=STEM_ANALYZER, 
    valid_size=VALID_SIZE, 
    max_seq_len=MAX_SEQ_LEN, 
    batch_size=BATCH_SIZE,
)
dm.setup('fit')

model = NSMCClassification()

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=CKPT_SAVE_PATH,
    filename='{epoch:02d}-{val_acc:.3f}',
    verbose=True,
    save_last=False,
    mode='max',
    save_top_k=1,
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
    auto_select_gpus=True,
    callbacks=[checkpoint_callback, early_stopping],
)
trainer.fit(model, dm)

trainer.test(ckpt_path='best')