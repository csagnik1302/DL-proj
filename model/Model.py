import torch.nn as nn
import torch
import vocab_creator
import input_creator
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import Adversary_Passes as Adv
import matplotlib.pyplot as plt
import math


#GPU use
device=torch.device('cpu')

if torch.cuda.is_available()==True:
    device=torch.device('cuda')



dataset = pd.read_csv(r"C:\project\DL project\Dataset_Creator\bangla_corpus\sentences.csv")

tokens=vocab_creator.tokenize(dataset['text'])
vocab=vocab_creator.vocab_creator(dataset['text'])

enc_tokens=input_creator.final_input(tokens).long()                      #embed wotks best with long
enc_author_tokens=input_creator.final_author_input(dataset["author"]).long()     
# author name needs to be encoded as long (int64) as cross entropy loss works with long data in pytorch
#(torch cant convert direct string lists into tensors)


#embedding
embed=nn.Embedding(num_embeddings=len(vocab),          #need to tune embedding_dim
                    embedding_dim=128,
                    padding_idx=vocab["<pad>"]).to(device)
                        




#mini-batch creation
mini_data1=TensorDataset(enc_tokens,enc_author_tokens)   
#wraps your tensor into a dataset object. Recommended for usage with dataloader. SInce raw tensors do not work well with dataloader. DataLoader works with datasets, not raw arrays. Especially useful when working with multuiple columns of data (input,label1,lebale2,.....)

loader=DataLoader(mini_data1,batch_size=64,shuffle=True)           # Tune batch size
#Dataloader splits data into mini batches     
#Shuffle =True othersie (see the dataset.csv) first model will train bibudhi, then next auithor,..... , leads to instability for discriminator


# Discriminator Freezing




loss_1_full=[]
loss_2_full=[]

loss_per_batch_1=0
loss_per_epoch_1=0

loss_per_batch_2=0
loss_per_epoch_2=0

Accuracy1=0
batch_count=0
Accuracy_epoch=0
Accuracy_all=[]

max_epoch=130

for epoch in range(max_epoch):

    for i in loader:
        
        logits_1,loss_1=Adv.Adversary_Pass1(i)

        loss_per_batch_1+=loss_1.item()           # Sums up losses for all batches in this epoch

        if epoch<30:                               # 30 cuz triend it out with l=0.0000001*(epoch/26)**2 normally, and it anyuways shows maxima at 30. This allows us to increase max achieved accuracy a little bit
            l=0                                    # Allows the encoder to train and understand author style patterns
        else:
            l=0.0000001*(epoch/26)**2      # Adversarial training, decooder slowly takes over # can be tuned   # so small value because even slight ly large value of lambda (e.g: 0.01 leading to quick convergence and thus this implies that )

        # After 25 epochs, the discriminator accuracy starts decreasing (this controls variance)
        # 0.00000001 controls maxima of accuracy (less this means more maxima)
        # Finding perfect tradeoff between this is the gaaaaammmeeeeee

        logits_2,loss_2=Adv.Adversary_Pass2(i,l)

        loss_per_batch_2+=loss_2.item()

        # ###### FOR DEBUGGING ####
        # logits_1,loss_1=Adv.Classification_pass(i)
        # loss_per_batch_1+=loss_1.item()           # Sums up losses for all batches in this epoch



        values1,indices1=torch.max(logits_1,dim=1)  # max() gives the max value, and index of max value (wrt dim) wrt dim. dim 1 means in each row, fine max value across dim 1 (columns)
        y_pred1=indices1.to(device)
        y_act1=i[1].to(device)
        correct1=torch.where(y_pred1==y_act1,1,0)
        sum1=torch.sum(correct1)
        count1=correct1.shape[0]
        Accuracy1+=sum1.item()
        batch_count+=count1

    loss_per_epoch_1=loss_per_batch_1/len(loader)
    loss_per_batch_1=0
    
    loss_per_epoch_2=loss_per_batch_2/len(loader)
    loss_per_batch_2=0

    Accuracy_epoch=Accuracy1/batch_count   # Basically total correct predictions/total batches count throughout the entire epoch
    Accuracy1=0
    batch_count=0

    loss_1_full.append(loss_per_epoch_1)       # Computes avergae of loss sum for a epoch (this is the output epoch loss)
    loss_2_full.append(loss_per_epoch_2)

    Accuracy_all.append(Accuracy_epoch)

    print(f"EPOCH {epoch}: PHASE-1: Loss: {loss_per_epoch_1}, Accuracy: {Accuracy_epoch*100}%, PHASE-2: Loss: {loss_per_epoch_2}")
    # print(f"EPOCH {epoch}: PHASE-1: Loss: {loss_per_epoch_1}, Accuracy: {Accuracy_epoch*100}%")
    # print(f"EPOCH {epoch}: CLASSIFICATION: Loss: {loss_per_epoch_1}, Accuracy: {Accuracy_epoch*100}%")

x=list(range(1,max_epoch+1))

plt.plot(x,loss_1_full)
plt.plot(x,loss_2_full)
plt.show()

plt.plot(x,Accuracy_all)
plt.show()



