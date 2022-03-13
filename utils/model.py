import torch
import torch.nn as nn

from model.transformer.module import Transformer
from model.bert_nmt.module import BertNMT




def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(model_name, config):
    if model_name == 'transformer':
        model = Transformer(config)
        model.apply(init_xavier)
    
    else:
        model = BertNMT(config)
        model.apply(init_xavier)
    

    model.to(config.device)
    print(f'{model_name} model has loaded. The model has {count_parameters(model):,} trainable parameters')

    return model