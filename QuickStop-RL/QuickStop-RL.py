#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import scipy.stats as st

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms
np.set_printoptions(threshold=np.inf)
from Dataset.NLST_Dataset_V19 import NLST_dataset

device = 'cuda' 

Transition = namedtuple('Transition',
                        ('Belief_N', 'Belief_P', 'index','ob_state', 'action','reward','Belief_next_N', 'Belief_next_P','ob_next_state','done','start','length','biospy','patient','pid'))


NLST = NLST_dataset()
NLST.initation_training()
NLST.initation_testing()
Memory = NLST.Train_Replay_memory
memory_test = NLST.Test_Replay_memory

B_I = NLST.B_I[0]  ### 
B_M = NLST.B_M[0]  ### 

M_B = NLST.M_B[0]
M_I = NLST.M_I[0]


print('Finish_Dataloader')

class Batch_Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return (x)


#n_actions = 3 # B,M,C
v_avtions = 1
policy_net = Batch_Net(4, 32, 32, v_avtions).to(device)
target_net = Batch_Net(4, 32, 32, v_avtions).to(device)


criterion = nn.SmoothL1Loss()
#criterion = nn.MSELoss()
#criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(policy_net.parameters(), lr=0.003)
#optimizer = optim.RMSprop(policy_net.parameters())
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


BATCH_SIZE = 8
GAMMA = 0.9


def optimize_model():
    if len(Memory) < BATCH_SIZE:
        return

    policy_net.train()


    transitions = Memory.sample(BATCH_SIZE)

    
    batch = Transition(*zip(*transitions))

    Belief_N_batch = torch.cat(batch.Belief_N).unsqueeze(1) # [32,1]
    index_batch = torch.cat(batch.index).unsqueeze(1) # [32,1]
    Belief_P_batch = torch.cat(batch.Belief_P).unsqueeze(1) # [32,1]
    ob_state_batch = torch.cat(batch.ob_state) # [32,3]   
    ob_state_batch[:,0] = (ob_state_batch[:,0]-2)/47
    action_batch = torch.cat(batch.action).unsqueeze(1)  # [32,1]
    reward_batch = torch.cat(batch.reward).unsqueeze(1)  # [32,1]


    Belief_next_N_batch = torch.cat(batch.Belief_next_N).unsqueeze(1) # [32,1]
    Belief_next_P_batch = torch.cat(batch.Belief_next_P).unsqueeze(1) # [32,1]
    ob_next_state_batch = torch.cat(batch.ob_next_state) # [32,3]
    ob_next_state_batch[:,0] = (ob_next_state_batch[:,0]-2)/47
    done_batch = torch.cat(batch.done).unsqueeze(1) # [32,1]

    '''
    print(Belief_N_batch.shape)
    print(Belief_P_batch.shape)
    print(ob_state_batch.shape)
    print(action_batch.shape)
    print(reward_batch.shape)
    print(Belief_next_N_batch.shape)
    print(Belief_next_P_batch.shape)
    print(ob_next_state_batch.shape)
    print(done_batch.shape)
    '''
   
    state_batch = torch.cat((Belief_N_batch,Belief_P_batch,ob_state_batch,index_batch),1)                       # [32,6]
    state_batch_next = torch.cat((Belief_next_N_batch,Belief_next_P_batch,ob_next_state_batch,index_batch),1)   # [32,6]

    optimizer.zero_grad()
    
    #state_action_values = policy_net(state_batch.to(device).float()).gather(1, action_batch.to(device)).squeeze(1)
    #next_state_values = policy_net(state_batch_next.to(device).float()).max(1)[0].detach()

    state_action_values = policy_net(state_batch.to(device).float())   # [32,1]  
    next_state_values = target_net(state_batch_next.to(device).float()).detach()  # [32,1] 

    stop_reward = torch.min(Belief_N_batch * M_B + M_I*(1-Belief_N_batch - Belief_P_batch), Belief_P_batch * B_M + B_I*(1-Belief_N_batch - Belief_P_batch)).to(device)   ### [32,1]  

    #expected_state_action_values = (next_state_values * GAMMA).mul(done_batch.to(device)) + reward_batch.to(device)
    expected_state_action_values = (next_state_values * GAMMA).mul(done_batch.to(device)) + reward_batch.to(device)  ### [32,1]
 

    #loss = criterion(state_action_values.float(), expected_state_action_values.float())
    loss = criterion(state_action_values.float(), torch.min(expected_state_action_values.float(),stop_reward.float()))
    loss.backward()
    optimizer.step()

    return loss



def Threshold_map():

    result_P = np.zeros((100,100,40))
    result_N = np.zeros((100,100,40))
    policy_net.eval()
    for Belief_N in range(100):
        for Belief_P in range(100-Belief_N):

            Belief_N_batch = torch.tensor([Belief_N/100]).unsqueeze(1).float()
            Belief_P_batch = torch.tensor([Belief_P/100]).unsqueeze(1).float()

            for D in range(40):
                diameter = D+5
                
                ob_state = np.zeros(1).astype(np.float32)
                ob_state[0] = (diameter-2)/47
                ob_state_batch = torch.tensor([ob_state])
                index = torch.tensor([0]).unsqueeze(1)

                state_batch = torch.cat((Belief_N_batch,Belief_P_batch,ob_state_batch,index),1).float()
                
                state_action_values = policy_net(state_batch.to(device)).detach().cpu() # [ 0.94695485 -0.7095688  -0.19343726]
                state_action_values = state_action_values[0] 
                
                #result_P[Belief_N,Belief_P,D] = state_action_values - Belief_N_batch * B_M
                #result_N[Belief_N,Belief_P,D] = state_action_values - Belief_P_batch * M_B

                result_P[Belief_N,Belief_P,D] = state_action_values
                result_N[Belief_N,Belief_P,D] = state_action_values

    return result_P,result_N




num_episodes = 100000000
time_step = np.zeros(num_episodes)
reward_array = np.zeros(num_episodes)
LOSS = np.zeros(num_episodes)
save_result = np.zeros((100000))
x = 0
OUTPUT = []
for i_episode in range(num_episodes):
    
    loss = optimize_model()
    LOSS[i_episode] = loss
    target_net.load_state_dict(policy_net.state_dict())
    #print(loss)
    
    if i_episode>100 and i_episode%5000 == 0 :

        lr_scheduler.step()
        print('Loss',np.mean(LOSS[i_episode-1000:i_episode]))
        threshold_result_P,threshold_result_N = Threshold_map()

        #Result = Eva_model(threshold_result_P,threshold_result_N)

        #np.save('OUTPUTTT/Result_OUTPUT_'+str('_%d' % i_episode)+'.npy',Result)
        #DF = pd.DataFrame(Result)
        #DF.to_csv('OUTPUTTT/AAA9'+str('_%d' % i_episode)+'.csv',index = False)         
        #Flow_chat(pd.read_csv('OUTPUTTT/AAA9'+str('_%d' % i_episode)+'.csv') )
        
        
        np.save('OUTPUTTT/OUTPUTP_'+str('_%d' % i_episode)+'.npy',threshold_result_P)
        np.save('OUTPUTTT/OUTPUTN_'+str('_%d' % i_episode)+'.npy',threshold_result_N)



