import vocab_creator
import pandas as pd
import torch
from copy import deepcopy
dataset=pd.read_csv(r"C:\project\DL project\model\sentences.csv")

vocab=vocab_creator.vocab_creator(dataset["text"])
tokens=vocab_creator.tokenize(dataset["text"])


def encoded_tokens(tokens1):
    enc=[]
    for i in tokens1:
        temp=[]
        for j in i:
            temp.append(vocab.get(j,vocab["<unk>"]))
        enc.append(temp)
    return enc

def padding(tokens2):                       #for padding
    tokens23=deepcopy(tokens2)
    len1=len(max(tokens23,key=len))
    for i in range(len(tokens23)):
        while len(tokens23[i])!=len1:
            tokens23[i].append(vocab["<pad>"])
    return tokens23


def final_input(tokens3):
    out=padding(encoded_tokens(tokens3))
    return torch.tensor(out)



