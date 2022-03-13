## Overview
Bert is one of the most important model in NLP.

This repo addresses how to apply BERT model to NMT Task.
The idea borrowed from <a href='https://arxiv.org/abs/2002.06823'>this paper</a>

<br>

## Directory
```
├── checkpoints        # Save model states and train_records
├── configs            # Configuration files for module
├── data               # Datasets will be stored in this directory
├── model              # Define Model Structures
│   ├── bert_nmt
│   └── transformer
├── scripts            # Contains Shell Scripts for download, build vocab
├── utils              # Contains helper fuctions for data, model, train, test
├── prepare_data.sh    # The Script carries out Download dataset, Build vocab and Tokenize processes
├── train.py
└── run.sh

```
