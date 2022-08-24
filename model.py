import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MTLModels(nn.Module):
    def __init__(self, transformer, drop_out, number_of_classes, heads):
        super(MTLModels, self).__init__()
        self.number_of_classes = number_of_classes
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(drop_out)
        self.heads = heads
        self.classifiers = dict()
        for head in self.heads:
            # self.classifiers[head] = nn.Linear(self.embedding_size * 2, self.number_of_classes[head]) #TODO: check to number of classes depend on the head/dataset
            self.classifiers[head] = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        
    def forward(self, iputs, head):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)

        return self.classifiers[head](drop)
    
    
    
    
# SB) I dont need to say what the model will receive as input
# class MTLModels(nn.Module):
#     def __init__(self, transformer, drop_out, number_of_classes):
#         super(MTLModels, self).__init__()
#         self.number_of_classes = number_of_classes
#         self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
#         self.transformer = AutoModel.from_pretrained(transformer)
#         self.dropout = nn.Dropout(drop_out)
#         self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        
#     def forward(self, iputs):
#         transformer_output  = self.transformer(**iputs)
#         mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
#         max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
#         cat = torch.cat((mean_pool,max_pool), 1)
#         drop = self.dropout(cat)

#         return self.classifier(drop)

    
# SA) I need to say what the model will receive as input
# class TransforomerModel(nn.Module):
#     def __init__(self, transformer, drop_out, number_of_classes):
#         super(TransforomerModel, self).__init__()
#         self.number_of_classes = number_of_classes
#         self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
#         self.transformer = AutoModel.from_pretrained(transformer)
#         self.dropout = nn.Dropout(drop_out)
#         self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        
#     def forward(self, ids, mask, token_type_ids):
#         transformer_output  = self.transformer(ids, attention_mask=mask, token_type_ids=token_type_ids)
#         mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
#         max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
#         cat = torch.cat((mean_pool,max_pool), 1)
#         drop = self.dropout(cat)

#         return self.classifier(drop)