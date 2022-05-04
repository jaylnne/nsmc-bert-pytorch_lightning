import torch
import argparse
from pytorch_lightning import Trainer, seed_everything

from model import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                       type=int,
                       default=42)
    parser.add_argument('--data_path',
                        type=str,
                        default='data',
                        help='where to prepare data')
    parser.add_argument('--num_gpus',
                       type=int,
                       default=-1,
                       help='number of available gpus')
    parser.add_argument('--mode',
                       type=str,
                       default='clean',
                       choices=['clean', 'only_korean'])
    parser.add_argument('--ckpt_path',
                       type=str,
                       help='checkpoint file path')
    parser.add_argument('--valid_size',
                       type=float,
                       default=0.1,
                       help='size of validation file')
    parser.add_argument('--max_seq_len',
                       type=int,
                       default=200,
                       help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help='batch size')
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    dm = NSMCDataModule(
        data_path=args.data_path, 
        mode=args.mode, 
        valid_size=args.valid_size, 
        max_seq_len=args.max_seq_len, 
        batch_size=args.batch_size,
    )
    dm.setup('test')

    model = NSMCClassification()

    trainer = Trainer(
        accelerator='gpu',
        strategy="ddp",
        devices=args.num_gpus,
        auto_select_gpus=True,
    )
    trainer.test(model, dm, ckpt_path=args.ckpt_path)
    
    
if __name__ == '__main__':
    main()