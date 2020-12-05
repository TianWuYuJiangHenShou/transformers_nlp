from transformers import BertConfig, BertTokenizer, BertModel
import torch
from soft_masked_bert import SoftMaskedBert


BERT_MODEL = '../../pretrain_models/bert_base_chinese'

class Augments():
    batch_size = 8
    num_workers = 8
    max_len = 128
    vocab_path = ''
    checkpoint = BERT_MODEL
    hidden_zie = 256
    rnn_layer = 1
    learning_rate = 2e-5
    hidden_size = 256
    max_grad_norm = 1.0


opt = Augments()
tokenizer = BertTokenizer.from_pretrained('../../pretrain_models/bert_base_chinese/vocab.txt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SoftMaskedBert(opt, tokenizer, device)
text = '去北京的酒店非用'
token = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(token)
mask =  [[1]*(len(ids))]
ids = torch.Tensor([ids]).long()
mask = torch.Tensor(mask).long()
p, log_probs = model(ids,mask)
# out = bert(ids)
print(p.shape)  #torch.Size([1, 8, 1])
print('***************************',)
print(log_probs.shape) # torch.Size([1, 8, 21128])
