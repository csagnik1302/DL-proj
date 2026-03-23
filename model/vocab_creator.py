from collections import Counter

def tokenize(data):
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
    "<eos>": 2
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


def encoded_tokens(tokens1):
    enc=[]
    for i in tokens1:
        temp=[]
        for j in i:
            temp.append(vocab.get(j,vocab["<unk>"]))
        enc.append(temp)
    return enc

  
    







