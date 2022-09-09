import config
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MTLModels(nn.Module):
    # def __init__(self, transformer, drop_out, heads, data_dict, model_name):
    def __init__(self, transformer, drop_out, heads, data_dict, model_name, num_efl, num_dfl):
        super(MTLModels, self).__init__()
        self.num_efl = num_efl
        self.num_dfl = num_dfl
        self.data_dict = data_dict
        self.model_name = model_name
        self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
        self.transformer = AutoModel.from_pretrained(transformer)
        self.dropout = nn.Dropout(drop_out)
        self.heads = heads
        self.classifiers = dict() #TODO: I should try a pytorch dict
        for head in self.heads:
            self.classifiers[head] = nn.Linear(self.embedding_size * 2, self.data_dict[head]['num_class']).to(config.DEVICE)
        
        if 'task-identification-vector' in config.MODELS[self.model_name]['encoder']['input']:
            self.encoder_feature_layers = nn.ModuleList()
            self.encoder_feature_layers.append(nn.ReLU(((self.embedding_size * 2) + len(self.heads))))
            self.encoder_feature_layers.append(nn.Linear(self.embedding_size * 2 + len(self.heads), self.embedding_size * 2))
            for _ in range(num_efl-1):
                self.encoder_feature_layers.append(nn.ReLU(self.embedding_size * 2))
                self.encoder_feature_layers.append(nn.Linear(self.embedding_size * 2 + len(self.heads), self.embedding_size * 2))
                #TODO: try to create the additional layer only in side the  for loop
                #TODO: think about the number of layers
    
    def forward(self, iputs, head):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        out = self.dropout(cat)
        
        if 'task-identification-vector' in config.MODELS[self.model_name]['encoder']['input']:
            task_ident_vector = [0] * len(self.heads)
            task_ident_vector[self.heads.index(head)] = 1
            task_ident_vector = torch.tensor([task_ident_vector] * out.shape[0], dtype=torch.long)
            out = torch.cat((out,task_ident_vector), 1)
            for layer in self.encoder_feature_layers:
                out = layer(out)

        return self.classifiers[head](out)
    
    
    
# class MTLModels(nn.Module):
#     def __init__(self, transformer, drop_out, heads, data_dict):
#         super(MTLModels, self).__init__()
#         self.data_dict = data_dict
#         self.embedding_size = AutoConfig.from_pretrained(transformer).hidden_size
#         self.transformer = AutoModel.from_pretrained(transformer)
#         self.dropout = nn.Dropout(drop_out)
#         self.heads = heads
#         self.classifiers = dict()
#         for head in self.heads:
#             self.classifiers[head] = nn.Linear(self.embedding_size * 2, self.data_dict[head]['num_class']).to(config.DEVICE)
            
#     def forward(self, iputs, head):
#         transformer_output  = self.transformer(**iputs)
#         mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
#         max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
#         cat = torch.cat((mean_pool,max_pool), 1)
#         drop = self.dropout(cat)

#         return self.classifiers[head](drop)