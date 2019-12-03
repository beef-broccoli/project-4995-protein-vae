#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ author: jason wang

Load trained VAE model
make inference on non-metal binding sequences with metal code requested

example usage: 
    python seq_to_metalseq.py -infile ./examples/seq2metalseq_example.txt -numout 10 -metal Fe

accepted metals:
    Fe, Zn, Ca, Na, Cu, Mg, Cd, Ni

"""

import utils # NEED utils.py

import torch
import torch.nn as nn

import numpy as np
import argparse
        
from sklearn.metrics import accuracy_score



batch_size=200
lr=5e-4
hidden_size=[512,256,128,16]
conv_size = [1, 64, 128, 1024]



# =============================================================================
# NEW SEQUENCE INFERENCE
# =============================================================================

def newMetalBinder(code,model,data):
    
    """
    Generates a new sequence based on a metal code.
    
    code is metal code
    model is trained model
    data is raw sequence
    """
    
    scores=[]
    model.eval()
    
    code = np.tile(code,(model.batch_size,1))
    x = np.tile(data[:3080],(model.batch_size,1))
    x = convert(x) # might be problematic
    X = torch.from_numpy(x).type(torch.FloatTensor)
    C = torch.from_numpy(code).type(torch.FloatTensor)

    x_sample, z_mu, z_var = model(X, C)
    
    
    len_aa=140*22
    y_label=np.argmax(x[:,:len_aa].reshape(batch_size,-1,22), axis=2)
    y_pred=np.argmax(x_sample[:,:len_aa].cpu().data.numpy().reshape(batch_size,-1,22), axis=2)
    for idx, row in enumerate(y_label):
        scores.append(accuracy_score(row[:np.argmax(row)],y_pred[idx][:np.argmax(row)]))
    print("Average Sequence Identity to Input: {0:.1f}%".format(np.mean(scores)*100))
    
    out_seqs=x_sample[:,:len_aa].cpu().data.numpy()
    for seq in out_seqs:
        seq = seq.reshape((seq.shape[1],seq.shape[0]))
        print(utils.vec_to_seq(seq))
        
    return


# =============================================================================
# model helper class and functions
# =============================================================================
    
def convert(x):
    m = len(x)
    n = len(x[0])
    y = np.empty([1, n, m])
    y[0] = np.transpose(x)
    y = np.transpose(y, (2,0,1))
    return y    


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height)


# =============================================================================
# MODEL
# =============================================================================
        
class feed_forward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, batch_size):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size

        self.encoder = nn.Sequential(
            nn.Conv1d(conv_size[0], conv_size[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_size[1], conv_size[2], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(98560, conv_size[3]),
            nn.ReLU()
        )   
 
        self.fc_mu = torch.nn.Linear(conv_size[3], hidden_size[3])
        self.fc_var = torch.nn.Linear(conv_size[3], hidden_size[3])
       
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[3]+8, conv_size[3]),
            nn.ReLU(),
            nn.Linear(1024, 98560),
            nn.ReLU(),
            Unflatten(128, 770),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_var(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn(self.batch_size, self.hidden_sizes[-1])
        return mu + torch.exp(log_var / 2) * eps
    
    def forward(self, x, code):

        mu, log_var = self.encode(x)
        z = self.sample_z(mu, nn.functional.softplus(log_var))
        z = torch.cat((z, code), 1)
        return self.decode(z), mu, log_var


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    
    # sort command line    
    parser = argparse.ArgumentParser()
    parser.add_argument("-infile", type=str, help="file with sequence", 
                        default="./examples/random.txt")# its either struc or nostruc
    parser.add_argument("-numout", type=int, help="number of sequences generated", 
                        default=10)
    parser.add_argument("-metal", type=str, help="one of: Fe, Zn, Ca, Na, Cu, Mg, Cd, Ni", 
                        default="Fe")
    
    args = parser.parse_args()
    args_dict = vars(args)
    
    # load model  
    X_dim=3088
    batch_size=args_dict["numout"]

    model = feed_forward(3088, hidden_size, batch_size)
    model.load_state_dict(torch.load('./model4-correct/checkpoint.pt', map_location=lambda storage, loc: storage))
    
    
    # first we read in the sequence from the 
    with open(args_dict["infile"],'r') as in_file:
        seq=in_file.readlines()
    
    # format the sequence so if it is a FASTA file then we turf the line with >
    for idx, line in enumerate(seq):
        seq[idx]=line.replace("\n","")
    
    seq_in=""
    for line in seq:
        if ">" in line:
            continue
        else:
            seq_in=seq_in+line
    
    # now have a string which is the sequence        
    seq_in_vec=utils.seq_to_vec(seq_in)
    
    print('Original sequence:')
    print(seq_in)
    print()
    
    # now we want to create the right metal code as supplied
    metals=['Fe', 'Zn', 'Ca', 'Na', 'Cu', 'Mg', 'Cd', 'Ni']
    metals_dict={}
    for idx, metal in enumerate(metals): metals_dict[metal]=idx
    try:
        code=np.zeros(8)
        code[metals_dict[args_dict["metal"]]]=1
    except:
        print("Please supply one of the correct 8 names for metals")
    
    newMetalBinder(code,model,seq_in_vec)
