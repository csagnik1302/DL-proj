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

if __name__=="__main__":
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



############################################################################################################################################################################

    loss_1_full=[]
    loss_2_full=[]
    total_loss_full=[]

    loss_per_batch_1=0
    loss_per_epoch_1=0

    loss_per_batch_2=0
    loss_per_epoch_2=0

    recon_loss_batch=0
    recon_loss_epoch=0

    total_loss_batch=0
    total_loss_epoch=0

    Accuracy1=0
    batch_count=0
    Accuracy_epoch=0
    Accuracy_all=[]

    max_epoch=400

    best_loss=float('inf')

    # Controls the ceiling of the gradient reversal strength.
    # DANN unscaled reaches ~0.29 at epoch 30/500 — far too aggressive (accuracy crashed).
    # Scaling by lambda_max keeps adversarial pressure gentle and stable.
    # Start with 0.05; raise if discriminator accuracy never meaningfully drops after epoch 30.
    # Lower if accuracy crashes again (drops >5% in a single epoch).
    k = 0.001

    for epoch in range(max_epoch):

        for i in loader:

            #################### Encoder + Discriminator #######################################
            
            logits_1,loss_1=Adv.Adversary_Pass1(i)

            loss_per_batch_1+=loss_1.item()           # Sums up losses for all batches in this epoch

            # if epoch<30:
            #     l=0                                    # Allows the encoder to train and understand author style patterns
            # else:
            #     # DANN schedule scaled by lambda_max.
            #     # Unscaled DANN: 0→~1.0. Multiplying by lambda_max keeps the ceiling at lambda_max.
            #     # At epoch 30/200: p=0.15 → raw≈0.46 → l≈0.023 (gentle start)
            #     # At epoch 200/200: p=1.0  → raw≈1.0  → l≈0.05  (full ceiling)
            #     p = epoch / 30
            #     l = k*p

            p = epoch/15
            l = k*p**2

            logits_2,loss_2,recon_loss,total_loss=Adv.Adversary_Pass2(i,l)

            loss_per_batch_2+=loss_2.item()
            recon_loss_batch+=recon_loss.item()
            total_loss_batch+=total_loss.item()

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

        recon_loss_epoch=recon_loss_batch/len(loader)
        recon_loss_batch=0

        total_loss_epoch=total_loss_batch/len(loader)
        total_loss_batch=0

        Accuracy_epoch=Accuracy1/batch_count   # Basically total correct predictions/total batches count throughout the entire epoch
        Accuracy1=0
        batch_count=0

        loss_1_full.append(loss_per_epoch_1)       # Computes avergae of loss sum for a epoch (this is the output epoch loss)
        loss_2_full.append(loss_per_epoch_2)
        total_loss_full.append(total_loss_epoch)

        Accuracy_all.append(Accuracy_epoch)

        print(f"EPOCH {epoch}: PHASE-1: Loss: {loss_per_epoch_1}, Accuracy: {Accuracy_epoch*100}%, PHASE-2: Loss: {loss_per_epoch_2}, Decoder Loss: {recon_loss_epoch}")
        # print(f"EPOCH {epoch}: Total Loss: {total_loss_epoch}")
        # print(f"EPOCH {epoch}: PHASE-1: Loss: {loss_per_epoch_1}, Accuracy: {Accuracy_epoch*100}%")
        # print(f"EPOCH {epoch}: CLASSIFICATION: Loss: {loss_per_epoch_1}, Accuracy: {Accuracy_epoch*100}%")

        # Save on reconstruction loss only.
        # total_loss includes the discriminator's cross-entropy (loss_2), which you *want*
        # to be HIGH (discriminator confused = good encoder). Minimising total_loss would
        # checkpoint the model where the discriminator is least confused — the opposite of
        # what adversarial training is trying to achieve.
        if recon_loss_epoch<best_loss:
            best_loss=recon_loss_epoch
            Adv.save_model(r"weights.pth")          # .pth is an extension that pytorch uses for saving and loading stuff internally



    x=list(range(1,max_epoch+1))

    # plt.plot(x,loss_1_full)
    # plt.plot(x,loss_2_full)
    # plt.show()

    plt.plot(x,Accuracy_all)
    plt.xlabel("Epochs")
    plt.xlabel("Accuracy")
    plt.show()

    plt.plot(x,total_loss_full)
    plt.show()