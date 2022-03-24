import torch
import torch.nn as nn

from model.module import BertNMT
from model.module_light import BertNMTLight




def init_xavier(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.xavier_uniform_(model.weight.data)



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(model_name, config):
    if model_name == 'bert_nmt':
        model = BertNMT(config)
        model.encoder.apply(init_xavier)
        model.decoder.apply(init_xavier)

    elif model_name == 'bert_nmt_light':
        model = BertNMTLight(config)
        model.encoder.apply(init_xavier)
        model.decoder.apply(init_xavier)
    

    model.to(config.device)
    print(f'{model_name} model has loaded. The model has {count_parameters(model):,} trainable parameters')

    return model