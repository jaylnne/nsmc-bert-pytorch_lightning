# Naver sentiment movie corpus(NSMC) BERT Pytorch Lightning

[[POST] ⚡pytorch lightning 으로 koBERT Fine-tuning 하기 - NSMC](https://velog.io/@jaylnne/Pytorch-Lightning-%EC%9C%BC%EB%A1%9C-koBERT-Fine-Tuning-%ED%95%B4%EB%B3%B4%EA%B8%B0-NSMC)

## Download pretrained model and tokenizer
```shell
$ python download_pretrained.py --save_path pretrained
```
- It will create `'./pretrained'` directory.
- It will download kobert pretrained model and tokenizer files below `'./pretrained'` directory.

## Train
```shell
# run example
python train.py
```
- arguments
    - seed: random seed number
    - data_path: where to prepare data
    - max_epoch: maximum number of epochs to train
    - num_gpus: number of available gpus (-1: all avauilable)
    - mode: train only korean data or not
    - save_path: where to save checkpoints files
    - valid_size: size of validation file
    - max_seq_len: number of available gpus
    - batch_size: batch size

## Test
```shell
# run example
python test.py --ckpt_path checkpoints/epoch=05-val_acc=0.897.ckpt
```
- arguments
    - ckpt_path: checkpoint file path which is execute test with
    - The rest is same with train.py
