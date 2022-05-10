import torch
import argparse
import numpy as np
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

    model = NSMCClassification()
    
    input_ids = torch.as_tensor(np.ones([1, 200]), dtype=int)
    attention_mask = torch.as_tensor(np.ones([1, 200]), dtype=int)
    token_type_ids = torch.as_tensor(np.ones([1, 200]), dtype=int)
    
    torch.onnx.export(model,
                      args=(input_ids, attention_mask, token_type_ids),
                      f='model.onnx',
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=11,
                      output_names=['output'],
    )
    

if __name__ == '__main__':
    
    main()