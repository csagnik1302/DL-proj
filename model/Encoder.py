import torch.nn as nn
import torch
import vocab_creator
import input_creator
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


#GPU use
device=torch.device('cpu')

if torch.cuda.is_available()==True:
    device=torch.device('cuda')



dataset = pd.read_csv(r"C:\project\DL project\model\sentences.csv")

tokens=vocab_creator.tokenize(dataset['text'])
vocab=vocab_creator.vocab_creator(dataset['text'])

enc_tokens=input_creator.final_input(tokens).long() #embed wotks best with long






#embedding
embed=nn.Embedding(num_embeddings=len(vocab),          #need to tune embedding_dim
                    embedding_dim=128,
                    padding_idx=vocab["<pad>"]).to(device)
                        






#mini-batch creation
mini_data1=TensorDataset(enc_tokens)   
#wraps your tensor into a dataset object. Recommended for usage with dataloader. SInce raw tensors do not work well with dataloader. DataLoader works with datasets, not raw arrays. Especially useful when working with multuiple columns of data (input,label1,lebale2,.....)

loader=DataLoader(mini_data1,batch_size=64,shuffle=True)           # Tune batch size
#Dataloader splits data into mini batches     
#Shuffle =True othersie (see the dataset.csv) first model will train bibudhi, then next auithor,..... , leads to instability for discriminator








# GRU Encoder

# output → hidden states at every time step

# t=1 → "He"       → h1  
# t=2 → "He is"    → h2  
# t=3 → "He is very" → h3  
# t=4 → full sentence → h4

# output = [h1, h2, h3, h4]


# hidden → hidden state at the end of the sequence

# hidden = forward → h4  
#          backward → h1 (reverse end)



gru=nn.GRU(input_size=128,hidden_size=128,batch_first=True,bidirectional=True).to(device)


hidden_all=[]

c=1

for i in loader:
    j=i[0].to(device)                 #Each item inside the loader is a tuple
    embed1=embed(j)

    #Forward pass (Back later after opposite grads)
    out,hidden=gru(embed1)

    hidden_both=torch.cat((hidden[0],hidden[1]),dim=1)    #concatenates hidden[0], hidden rep when forward reading, and hidden[1], for backward reading, each has dim(2,64,128), here we are concatenating along dim=1(col) so dim= (64,256)

    print(f"processing iteration - {c}")
    hidden_all.append(hidden_both.detach().cpu())

    c+=1

print(hidden_all[0].shape)


