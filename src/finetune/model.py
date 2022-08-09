from transformers import AutoModel
from torch import nn
import torch
class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        self.bert_model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.out = nn.Linear(768, 2)
        
    def forward(self,ids,mask):
        _,emb = self.bert_model(ids,attention_mask=mask, return_dict=False)
        out = self.out(emb)
    
        return out, emb
