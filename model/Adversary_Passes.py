import torch.nn as nn
import torch
import vocab_creator
import input_creator
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import itertools
#GPU use
device=torch.device('cpu')

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








# Encoder + Discriminator

# output → hidden states at every time step

# t=1 → "He"       → h1  
# t=2 → "He is"    → h2  
# t=3 → "He is very" → h3  
# t=4 → full sentence → h4

# output = [h1, h2, h3, h4]


# hidden → hidden state at the end of the sequence

# hidden = forward → h4  
#          backward → h1 (reverse end)



# GRU Architecture Input
gru_input=embed.embedding_dim

# GRU Architecture
gru=nn.GRU(input_size=gru_input,hidden_size=130,batch_first=True,bidirectional=True).to(device)

gru_output=260
class_count=5


# DISCRIMINATOR Input:

# CAN ALSO TUNE NUMBER OF LAYERS IN DISCRIMINATOR (CURRENTLY ITS 2)

l1_input=260
l1_output=120                   # Need to tune this

l2_input=120
l2_output=60                   # Need to tune this

# iN 2ND lAYER ReLU does not change shape (basically neuron count in layer 2 is same as layer 1)
# So 260 input -> 120 neuron Layer 1 -> 120 neuron Layer 2 (ReLU)

l3_input=60
l3_output=5                    #Since total number of labels =5


# Layer-1 (Linear) (Needed to learn more features)
l1_linear=nn.Linear(in_features=l1_input,out_features=l1_output,bias=True).to(device)     
# (ReLU)  (To introduce Non-Linearity)
l1_relu=nn.LeakyReLU()


# Layer-2 (Linear) (Needed to learn more features)
l2_linear=nn.Linear(in_features=l2_input,out_features=l2_output,bias=True).to(device)     
# (ReLU)  (To introduce Non-Linearity)
l2_relu=nn.LeakyReLU()


# Layer-3 (Linear) (Output Layer) (To produce logits) (Probabilities are explicity produced in nn.crossentropy so we did not put softmax in the output layer, otherwise we would have only done softmax twice which will give wrong result)
l3_linear=nn.Linear(in_features=l3_input,out_features=l3_output,bias=True).to(device)


# Cross-Entropy Loss:
cross_loss=nn.CrossEntropyLoss().to(device)


# Define Optimizer (for optimization of parameter)

# LOGIC: Encoder: Trained to give style invariant (Confused) representation 
# Discriminator: Trained to correctly classify, but due to invarient input from gru, gives incorrect one.
#  2 optimizers are needed to introuce an adversarial training approach, where one looks at other to update (imagine a chess game, one person looks at others moves then thinks, and then plays his move) 

parameters_en=itertools.chain(embed.parameters(),gru.parameters())      # Only updates the gru parameters
parameters_dis=itertools.chain(l1_linear.parameters(),l2_linear.parameters(),l3_linear.parameters())     # Only updates the discriminator parameters  # ReLU has no parameters

parameters=itertools.chain(embed.parameters(),gru.parameters(),l1_linear.parameters(),l2_linear.parameters(),l3_linear.parameters())  # For plain gru + discriminator training (debug purpose) (non-adversarial)

# Used itertools.chain because simple list is only a list of iterables ([ iterator1, iterator2, iterator3 ]). 
# Each iterbale contains parameter data inside, so its simply a nested iterable of parameters ([ iterator1, iterator2, iterator3 ]) -> ([p1,p2,....],[p3,p4,...],...).
# Using chaoin converts this nested structure into a combined iterable of parameter upfront ([p1, p2, p3, p4, p5, ...]).

optimizer_en=torch.optim.Adam(parameters_en,lr=0.0001)     # Optimizer for Encoder training  # TUNE  LR
optimizer_dis=torch.optim.Adam(parameters_dis,lr=0.01)     # Optimizer for Discriminator training  # TUNE  LR
optimizer=torch.optim.Adam(parameters,lr=0.001)  # For plain gru + discriminator training (debug purpose) (non-adversarial) 

# Optimizer in pytorch takes the parameter input of all architectures that you have built and that needs to be updated
# through this optimizer object we can optimize parameters during backprop now



def freeze_discriminator():
        for i in l1_linear.parameters():
                i.requires_grad=False
        for j in l2_linear.parameters():
                j.requires_grad=False
        for k in l3_linear.parameters():
                k.requires_grad=False

def unfreeze_discriminator():
        for i in l1_linear.parameters():
                i.requires_grad=True
        for j in l2_linear.parameters():
                j.requires_grad=True
        for k in l3_linear.parameters():
                k.requires_grad=True




def Classification_pass(input):

        i_sent=input[0].to(device)                 #Each item inside the loader is a tuple, which looks like (Embedding of a sentence,author of that sentence)
        i_auth=input[1].to(device) 

        embed1=embed(i_sent)

        #Forward pass (Back later after opposite grads)
        out,hidden=gru(embed1)

        hidden_both=torch.cat((hidden[0],hidden[1]),dim=1)    #(DID THIS SINCE THIS IS WHAT DISCRIMINATOR EXPECTS AS INPUT) concatenates hidden[0], hidden rep when forward reading, and hidden[1], for backward reading, each has dim(2,64,128), here we are concatenating along dim=1(col) so dim= (64,256)


        # DISCRIMINATOR ARCHITECTURE:

        # Forward Propagation

        # Layer-1 (Linear) (Needed to learn more features) 
        l1_out=l1_linear(hidden_both)
        l1_relu_out=l1_relu(l1_out)
        
        # Layer-2 (Linear) (Needed to learn more features) 
        l2_out=l2_linear(l1_relu_out)
        l2_relu_out=l2_relu(l2_out)

        # Layer-3 (Linear) (Output Layer) (To produce logits) (Probabilities are explicity produced in nn.crossentropy so we did not put softmax in the output layer, otherwise we would have only done softmax twice which will give wrong result)
        l3_out=l3_linear(l2_relu_out)


        # Cross-Entropy Loss
        loss=cross_loss(l3_out,i_auth)

        # Backward Propagation
        optimizer.zero_grad()            
        # (Pytorch has a habit of accumulating previously computed gradients. This tackles that problem) 
        # Sets all previously computed gradients to 0, otherwise gradients of past batch will influence gradient computation of current batch, leading to inaccuracies

        loss.backward()

        optimizer.step()                 # Perform a single optimization step to update parameter.

        return l3_out,loss




def Adversary_Pass1(input):

        unfreeze_discriminator()            # To ensure discriminators gradients are being computed

        i_sent=input[0].to(device)                 #Each item inside the loader is a tuple, which looks like (Embedding of a sentence,author of that sentence)
        i_auth=input[1].to(device) 

        embed1=embed(i_sent)

        #Forward pass (Back later after opposite grads)
        out,hidden=gru(embed1)

        hidden_both=torch.cat((hidden[0],hidden[1]),dim=1)    #(DID THIS SINCE THIS IS WHAT DISCRIMINATOR EXPECTS AS INPUT) concatenates hidden[0], hidden rep when forward reading, and hidden[1], for backward reading, each has dim(2,64,128), here we are concatenating along dim=1(col) so dim= (64,256)

        hidden_both=hidden_both.detach()

        #.detach() prevents the gradients from flowing during backpropagation and thus prevents the encoder gradients from being generated (can thus interfere) when applying loss.backward()
        # NOTE: No problem during forward pass as .detach() only affects backward pass
        # NOTE: This is only for backward_pass() step which generates gradients. It does not update parameters so we are NOT DOING THIS TO PREVENT PARAMETER UPDATION BUT RATHER TO PREVENT GRADIENT GENERATION BY BLOCKING THE FLOW (SInce it can otherwise interfere unknowingly)

        # DISCRIMINATOR ARCHITECTURE:

        # Forward Propagation

        # Layer-1 (Linear) (Needed to learn more features) 
        l1_out=l1_linear(hidden_both)
        l1_relu_out=l1_relu(l1_out)
        
        # Layer-2 (Linear) (Needed to learn more features) 
        l2_out=l2_linear(l1_relu_out)
        l2_relu_out=l2_relu(l2_out)

        # Layer-3 (Linear) (Output Layer) (To produce logits) (Probabilities are explicity produced in nn.crossentropy so we did not put softmax in the output layer, otherwise we would have only done softmax twice which will give wrong result)
        l3_out=l3_linear(l2_relu_out)


        # Cross-Entropy Loss
        loss=cross_loss(l3_out,i_auth)

        # Backward Propagation
        optimizer_dis.zero_grad()            
        # (Pytorch has a habit of accumulating previously computed gradients. This tackles that problem) 
        # Sets all previously computed gradients to 0, otherwise gradients of past batch will influence gradient computation of current batch, leading to inaccuracies

        loss.backward()

        optimizer_dis.step()                 # Perform a single optimization step to update parameter.

        return l3_out,loss




def Adversary_Pass2(input,lam):

        freeze_discriminator()                  # To ensure gradients of discriminator are not being computed (precuationary measure)


        i_sent=input[0].to(device)                 #Each item inside the loader is a tuple, which looks like (Embedding of a sentence,author of that sentence)
        i_auth=input[1].to(device) 

        embed1=embed(i_sent)

        #Forward pass (Back later after opposite grads)
        out,hidden=gru(embed1)

        hidden_both=torch.cat((hidden[0],hidden[1]),dim=1)    #(DID THIS SINCE THIS IS WHAT DISCRIMINATOR EXPECTS AS INPUT) concatenates hidden[0], hidden rep when forward reading, and hidden[1], for backward reading, each has dim(2,64,128), here we are concatenating along dim=1(col) so dim= (64,256)



        # GRADIENT REVERSAL LAYER
        l=lam                      # Lambda value (TUNE IT)
        hidden_both.register_hook(lambda grad:-l*grad)           # Whenever dL/d(hidden_both) is computed, the register.hook method through a backward hook overwrites this with -l*dL/D(hidden_both)

        # .register_hook() Registers a backward hook.
        # The hook will be called every time a gradient with respect to the Tensor is computed. The hook should have the following signature:
        # hook(grad) -> Tensor or None
        # The hook should not modify its argument, but it can optionally return a new gradient which will be used in place of grad




        # DISCRIMINATOR ARCHITECTURE:

        # Not .detach() used since gru gradient cmomputation would require discriminator cgrads computatinjon (cuz chain rule). Also optimzer_en is not oging to update discriminator parameters so we are safe (ACTUALLY SAME FOR adversary 1 as well, but we still used regardless to prevent any problems and just to follow logic)

        # Forward Propagation

        # Layer-1 (Linear) (Needed to learn more features) 
        l1_out=l1_linear(hidden_both)
        l1_relu_out=l1_relu(l1_out)
        
        # Layer-2 (Linear) (Needed to learn more features) 
        l2_out=l2_linear(l1_relu_out)
        l2_relu_out=l2_relu(l2_out)

        # Layer-3 (Linear) (Output Layer) (To produce logits) (Probabilities are explicity produced in nn.crossentropy so we did not put softmax in the output layer, otherwise we would have only done softmax twice which will give wrong result)
        l3_out=l3_linear(l2_relu_out)


        # Cross-Entropy Loss
        loss=cross_loss(l3_out,i_auth)

        # Backward Propagation
        optimizer_en.zero_grad()            
        # (Pytorch has a habit of accumulating previously computed gradients. This tackles that problem) 
        # Sets all previously computed gradients to 0, otherwise gradients of past batch will influence gradient computation of current batch, leading to inaccuracies

        loss.backward()

        optimizer_en.step()                 # Perform a single optimization step to update encoder parameter.

        return l3_out,loss