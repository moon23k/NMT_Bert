import copy
import torch
import torch.nn as nn
from layer import EncoderLayer, DecoderLayer, get_clones
from transformers import BertModel



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.n_layers = config.n_layers
        self.layer = EncoderLayer(config)


    def forward(self, src, bert_out, src_mask):
        for layer in range(self.n_layers):
            src = self.layer(src, bert_out, src_mask)

        return src




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.n_layers = config.n_layers
        self.layer = DecoderLayer(config)


    def forward(self, memory, trg, bert_out, src_mask, trg_mask):
        for layer in range(self.n_layer):
            trg, attn = self.layer(memory, trg, bert_out, src_mask, trg_mask)
        
        return trg, attn




class BertNMTLight(nn.Module):
    def __init__(self, config):
        super(BertNMT, self).__init__()

        self.bert = BertModel.from_pretrained(config.bert_model)
        self.embedding = self.bert.embeddings

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.fc_out = nn.Linear(config.hidden_dim, config.output_dim)
        self.device = config.device


    def forward(self, src, trg, src_mask, trg_mask):
        
        bert_out = self.bert(src).last_hidden_state
        src, trg = self.embedding(src), self.embedding(trg) 
        
        enc_out = self.encoder(src, bert_out, src_mask)
        dec_out, _ = self.decoder(enc_out, trg, bert_out, src_mask, trg_mask)

        out = self.fc_out(dec_out)

        return out
