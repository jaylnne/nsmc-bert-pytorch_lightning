import torch
from pytorch_lightning import Trainer, seed_everything

from model import *


seed_everything(42, workers=True)

AVAIL_GPUS = -1
STEM_ANALYZER = 'mecab'
CKPT_PATH = 'checkpoints/epoch=01-val_acc=0.866.ckpt'
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
dm.setup('test')

model = NSMCClassification()

trainer = Trainer(
    accelerator='gpu',
    strategy="ddp",
    devices=AVAIL_GPUS,
    auto_select_gpus=True,
)
trainer.test(model, dm, ckpt_path=CKPT_PATH)