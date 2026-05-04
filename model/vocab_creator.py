from collections import Counter
import pandas as pd
import sentencepiece

data=pd.read_csv(r"C:\project\DL project\Dataset_Creator\bangla_corpus\sentences.csv")

def tokenize(data):

    ### For Normal (Word Level Tokenization) ##############
    token_set=[]
    for i in data:
        temp=i.split()
        temp.append("<eos>")
        token_set.append(temp)

    return token_set


def vocab(data1):

    vocab = {
    "<pad>": 0,
    "<unk>": 1,
    "<eos>": 2,
    "<sos>": 3
    }   

    count=Counter()
    for i in data1:
        count.update(i)

    for i,j in count.items():
        if j>5 and i not in vocab:
            vocab[i]=len(vocab)
    
    return vocab


def vocab_creator(data2):
    return vocab(tokenize(data2))


if __name__=="__main__":
    print(tokenize(data["text"])[:5])


  
    







