import vocab_creator
import pandas as pd
import torch
from copy import deepcopy
dataset=pd.read_csv(r"C:\project\DL project\Dataset_Creator\bangla_corpus\sentences.csv")

vocab=vocab_creator.vocab_creator(dataset["text"])
tokens=vocab_creator.tokenize(dataset["text"])


def encoded_tokens(tokens1):
    enc = []
    for i in tokens1:
        temp = [vocab["<sos>"]]   # prepend <sos>
        for j in i:
            temp.append(vocab.get(j, vocab["<unk>"]))
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



# For author encoding

def final_author_input(data):
    enc=[]
    for i in data:
        if i=="Bibhutibhushan Bandopadhyay".lower():
            enc.append(0)                                 #Cross Entropy needs class labels to start from 0 in pytorch
        elif i=="Rabindranath Tagore".lower():
            enc.append(1)
        elif i=="Sarat Chandra Chattopadhyay".lower():
            enc.append(2)
        elif i=="Satyajit Ray".lower():
            enc.append(3)
        elif i=="Sunil Gangopadhay".lower():
            enc.append(4)
    return torch.tensor(enc)
    


if __name__=="__main__":       #just for testing
    # print(tokens[0])
    print(final_input(dataset["text"]).size()[-1])




