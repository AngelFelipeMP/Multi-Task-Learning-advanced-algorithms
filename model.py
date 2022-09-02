import config
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MTLModels(nn.Module):
    def __init__(self, transformer, drop_out, heads, data_dict):
        super(MTLModels, self).__init__()
        self.data_dict = data_dict
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(drop_out)
        self.heads = heads
        self.classifiers = dict()
        for head in self.heads:
            self.classifiers[head] = nn.Linear(self.embedding_size * 2, self.data_dict[head]['num_class']).to(config.DEVICE)
            
    def forward(self, iputs, head):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)

        return self.classifiers[head](drop)