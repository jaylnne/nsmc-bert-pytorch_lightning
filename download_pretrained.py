import argparse
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--save_path',
                   type=str,
                   default='pretrained',
                   help='where to save pretrained model and tokenizer')
args = parser.parse_args()
    

model = BertModel.from_pretrained('skt/kobert-base-v1', output_attentions=True)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)