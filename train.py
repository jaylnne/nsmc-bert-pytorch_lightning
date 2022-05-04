import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *
from preprocessing import generate_preprocessed


parser = argparse.ArgumentParser()
parser.add_argument('--seed',
                   type=int,
                   default=42)
parser.add_argument('--data_path',
                    type=str,
                    help='where to prepare data')
parser.add_argument('--max_epoch',
                   type=int,
                   help='maximum number of epochs to train')
parser.add_argument('--num_gpus',
                   type=int,
                   default=-1,
                   help='number of available gpus')
parser.add_argument('--mode',
                   type=str,
                   default='clean',
                   choices=['clean', 'only_korean'])
parser.add_argument('--save_path',
                   type=str,
                   help='where to save checkpoint files')
parser.add_argument('--valid_size',
                   type=float,
                   default=0.1,
                   help='size of validation file')
parser.add_argument('--max_seq_len',
                   type=int,
                   help='maximum length of input sequence data')
parser.add_argument('--batch_size',
                   type=int,
                   help='batch size')
args = parser.parse_args()


seed_everything(args.seed, workers=True)

dm = NSMCDataModule(
    data_dir='./data', 
    stem_analyzer=args.mode, 
    valid_size=args.valid_size, 
    max_seq_len=args.max_seq_len, 
    batch_size=args.batch_size,
)
dm.prepare_data(args.data_path)
generate_preprocessed(args.data_path)
dm.setup('fit')

model = NSMCClassification()

checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath=args.save_path,
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
    max_epochs=args.max_epoch,
    accelerator='gpu',
    strategy="ddp",
    devices=args.num_gpus,
    auto_select_gpus=True,
    callbacks=[checkpoint_callback, early_stopping],
)
trainer.fit(model, dm)

trainer.test(ckpt_path='best')