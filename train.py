import os
import time
import math
import yaml
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data import get_dataloader
from utils.model import load_model
from utils.train import train_epoch, valid_epoch, epoch_time
from utils.scheduler import get_scheduler





class Config(object):
    def __init__(self, args):
        
        files = [f"configs/{file}.yaml" for file in [args.model, 'train']]

        for file in files:
            with open(file, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            for p in params.items():
                setattr(self, p[0], p[1])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_valid_loss = float('inf')
        self.learning_rate = 5e-4

    def print_attr(self):
        for attribute, value in self.__dict__.items():
            print(attribute, ': ', value)





def run(args, config):
    #set checkpoint, record path
    chk_dir = f"checkpoints/{args.model}/{args.data}/"
    os.makedirs(chk_dir, exist_ok=True)
    
    chk_file = "train_states.pt"
    record_file = "train_record.json"

    
    chk_path = os.path.join(chk_dir, chk_file)
    record_path = os.path.join(chk_dir, record_file)
    
    
    #define training record dict
    train_record = defaultdict(list)
        
    #get dataloader from chosen dataset
    train_dataloader = get_dataloader(args.data, 'train', config.batch_size)
    valid_dataloader = get_dataloader(args.data, 'valid', config.batch_size)
    
    
    #load model, criterion, optimizer, scheduler
    model = load_model(args.model, config)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    

    record_time = time.time()
    #train loop
    for epoch in range(config.n_epochs):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, config.clip, config.device)
        valid_loss = valid_epoch(model, valid_dataloader, criterion, config.device)
        

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)


        #save training records
        train_record['epoch'].append(epoch+1)
        train_record['train_loss'].append(train_loss)
        train_record['valid_loss'].append(valid_loss)
        train_record['lr'].append(optimizer.param_groups[0]['lr'])


        #save best model
        if valid_loss < config.best_valid_loss:
            config.best_valid_loss = valid_loss
            torch.save({'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'train_loss': train_loss,
                        'valid_loss': valid_loss}, chk_path)

        print(f"Epoch: {epoch + 1} | Time: {epoch_mins}m {epoch_secs}s")
        print(f'\tTrain Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')


    train_mins, train_secs = epoch_time(record_time, time.time())
    train_record['train_time'].append(f"{train_mins}min {train_secs}sec")



    #save ppl score to train_record
    for (train_loss, valid_loss) in zip(train_record['train_loss'], train_record['valid_loss']):
        train_ppl = math.exp(train_loss)
        valid_ppl = math.exp(valid_loss)

        train_record['train_ppl'].append(round(train_ppl, 2))
        train_record['valid_ppl'].append(round(valid_ppl, 2))


    #save train_record to json file
    with open(record_path, 'w') as fp:
        json.dump(train_record, fp)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-data', required=True)
    args = parser.parse_args()
    
    assert args.model in ['transformer', 'bert_base', 'bert_large', 'distil_bert', 'albert', 'roberta']
    assert args.data in ['wmt', 'iwslt', 'multi30k']
    
    config = Config(args)
    
    run(args, config)