import math
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import scipy.stats as st
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms


Transition = namedtuple('Transition',
                        ('Belief_N', 'Belief_P', 'index','ob_state', 'action','reward','Belief_next_N', 'Belief_next_P','ob_next_state','done','start','length','biospy','patient','pid'))


class ReplayMemory(object):

    def __init__(self):
        self.memory = deque([])

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NLST_dataset():
    def __init__(self):

        self.Train_Replay_memory = ReplayMemory()
        self.Test_Replay_memory = ReplayMemory()
    
        self.state_transition = np.zeros((3,3)).astype(np.float32)   # 522 vs 1429
        self.state_transition[0,0] = 1
        self.state_transition[1,0] = 0.3
        self.state_transition[1,1] = 0.5
        self.state_transition[1,2] = 0.2
        self.state_transition[2,2] = 1

        self.r = 0.034
        self.std_n = 3

        self.B_CT = [5]
        self.M_CT = [5]


        self.B_M = [15]
        self.B_I = [6.1]
     
        

        self.M_B = [15]
        self.M_I = [6.1]
        
   
    def loss_Geo(self,z,time):
    
        r = 0.034

        V = (z*z*z*np.pi)/6
        V = V + V*13*r*np.exp(-1*r)
        V = V/(np.pi/6)
        V = pow(V,1/3)

        dv = V-z
        k = dv*time/5
        return k

    def Model_Gom(self,D):
        D = np.float32(D)
        V = D*D*D*np.pi/6
        V =  V + V*13*self.r * np.exp(-self.r*1)
        V =  V + V*13*self.r * np.exp(-self.r*1)
        V =  V + V*13*self.r * np.exp(-self.r*1)
        #V =  V + V*13*self.r * np.exp(-self.r*1)
        #print(pow(V/(np.pi/6), 1/3))
        return pow(V/(np.pi/6), 1/3)
    
    def Calculate_Prob_alpha(self,Z,Z_next):
        
        prob_1 = st.norm.pdf(Z_next[0], loc=Z[0], scale=self.std_n/3)   ### value mean scale  ### 
        
        norm = 0
        for zzz in range(28):
            norm = norm + st.norm.pdf(zzz, loc=Z[0], scale=self.std_n/3)
        
        prob_1 = prob_1/norm
        #prob_2 = self.Ob3_alpha[int(Z[1]),int(Z_next[1])]

        return prob_1,0

    def Calculate_Prob_beta(self,Z,Z_next):
        
        prob_1 = st.norm.pdf(Z_next[0], loc=Z[0], scale=self.std_n)  ### value mean scale  ### 

        #prob_2 = self.Ob3_beta[int(Z[1]),int(Z_next[1])]
        norm = 0
        for zzz in range(28):
            norm = norm + st.norm.pdf(zzz, loc=Z[0], scale=self.std_n)
        
        prob_1 = prob_1/norm


        return prob_1,0

    def Calculate_Prob_gamma(self,Z,Z_next,last_CT):
       
        Z = np.array(Z)
        Z_next = np.array(Z_next)

        #Z[1] = self.Model_Gom(Z[1]) + np.random.normal(loc=0, scale=2, size=1)
        Z[0] = self.Model_Gom(Z[0])
        prob_1 = st.norm.pdf(Z_next[0], loc=Z[0], scale=2)  + 0.00001

        
        norm = 0
        for zzz in range(28):
            norm = norm + st.norm.pdf(zzz, loc=Z[0], scale=2)
        
        prob_1 = prob_1/norm

        #prob_2 = self.Ob3_gamma[int(Z[1]),int(Z_next[1])] 
        return prob_1,0
    
    def Update_belief(self,Pi_N,Pi_P,A,ALPHA,BATE,GAMMA):

        Pi_E = (1-Pi_N-Pi_P)
    
        Pi_N_hat = Pi_N + self.state_transition[1,0]*Pi_E
        Pi_P_hat = Pi_P + self.state_transition[1,2]*Pi_E

        Pi_E_hat = (1-Pi_N_hat-Pi_P_hat)
        All = Pi_E_hat * ALPHA[A] + Pi_N_hat * BATE[A] + Pi_P_hat*GAMMA[A]
        Pi_N = (Pi_N_hat*BATE[A]/All)
        Pi_P = (Pi_P_hat*GAMMA[A]/All)

        return Pi_N.astype(np.float32), Pi_P.astype(np.float32)  


    def initation_training(self):

        Data_whole = pd.read_excel("All_Training_2022.xlsx")
        #ids_whole = Data_whole['unique_ids'].unique()
        unique_ids_train = np.load('Training_selected_4-30_v2.npy')
        #unique_ids_IN = np.load('Testing_selected_4-30_v2-1.npy')

        Data_part = Data_whole.loc[Data_whole['unique_ids'].isin(unique_ids_train)]
        Data_part.index = range(len(Data_part))
        
        Belief_N_previous = -1
        Belief_P_previous = -1

        index = 0

        
        for row in range(len(Data_part)):
    
            if Belief_N_previous < 0:
                #Belief_P = (Data_part.loc[row,'probability']/100).astype(np.float32) 
                Belief_P = (Data_part.loc[row,'value_N']).astype(np.float32) 
                Belief_N = ((1 - Belief_P)).astype(np.float32)
                #Belief_N = 0.01
                #Belief_N = Belief_P.astype(np.float32) 
                start = 1
                index = 0
            else:
                Belief_N = Belief_N_previous
                Belief_P = Belief_P_previous
                index = index + 1
                start = 0

            #ob_state = np.zeros(4).astype(np.float32)
            #ob_state[0] = Data_part.loc[row,'SCT_LONG_DIA']
            #ob_state[2] = Data_part.loc[row,'SCT_PRE_ATT_b']
            #ob_state[3] = Data_part.loc[row,'SCT_MARGINS_b']
            ob_state = np.zeros(1).astype(np.float32)
            ob_state[0] = Data_part.loc[row,'SCT_LONG_DIA']              

            patient = Data_part.loc[row,'can_scr']
            traj_id = Data_part.loc[row,'unique_ids']
            trajectory = Data_part.loc[Data_part['unique_ids'] == traj_id]                
            length = len(trajectory)
            
            if row != len(Data_part)-1:            
                #ob_next_state = np.zeros(4).astype(np.float32)
                #ob_next_state[0] = Data_part.loc[row+1,'SCT_LONG_DIA'] 
                #ob_next_state[2] = Data_part.loc[row+1,'SCT_PRE_ATT_b']
                #ob_next_state[3] = Data_part.loc[row+1,'SCT_MARGINS_b'] 
                ob_next_state = np.zeros(1).astype(np.float32)
                ob_next_state[0] = Data_part.loc[row+1,'SCT_LONG_DIA']            
            else:
                ob_next_state = copy.deepcopy(ob_state)
            


            
            #####  A  End of Trajectory ###################################
            if row == len(Data_part)-1 or Data_part.loc[row,'unique_ids'] != Data_part.loc[row+1,'unique_ids']:

                Belief_N_previous = -1
                Belief_P_previous = -1

                #done_0 = [0]
                done_1 = [1]
                #biospy = [0]


                '''
                Diagonsis1 = [0]
                if patient == 1:
                    reward1 = self.M_CT + self.loss_Geo(ob_state,index+1)
                elif patient == 0:
                    reward1 = self.B_CT 
                '''
                
                if patient == 1:
                    Diagonsis = [2]
                    reward = (1-Belief_P-Belief_N) * self.M_I[0] + Belief_N * self.M_B[0]
                    #invers_Diagonsis = [1]
                    #invers_reward = Belief_P * self.B_M + (1-Belief_P-Belief_N) * self.B_I

                elif patient == 0:
                    Diagonsis = [1]
                    reward = (1-Belief_P-Belief_N) * self.B_I[0] + Belief_P * self.B_M[0]
                    #invers_Diagonsis = [2]
                    #invers_reward = Belief_P * self.B_M + (1-Belief_P-Belief_N) * self.B_I
                


                
                ob_next_state = copy.deepcopy(ob_state)
                Belief_next_N = copy.deepcopy(Belief_N)
                Belief_next_P = copy.deepcopy(Belief_P)

                '''
                self.Train_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(Diagonsis), torch.tensor(reward),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]),
                                            torch.tensor(done_0), torch.tensor([start]),torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
            
                self.Train_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(invers_Diagonsis), torch.tensor(invers_reward),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done_0), torch.tensor([start]), torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
                '''
                self.Train_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(Diagonsis), torch.tensor([reward]),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done_1), torch.tensor([start]), torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
                  
            #####  C  ###################################
            elif Data_part.loc[row,'unique_ids'] == Data_part.loc[row+1,'unique_ids']:
                
                prob_1,prob_2 = self.Calculate_Prob_alpha(ob_state,ob_next_state)
                ALPHA = [prob_1,prob_2]
                prob_1,prob_2 = self.Calculate_Prob_beta(ob_state,ob_next_state)
                BATE = [prob_1,prob_2]
                prob_1,prob_2 = self.Calculate_Prob_gamma(ob_state,ob_next_state,1)
                GAMMA = [prob_1,prob_2]
                Belief_next_N, Belief_next_P = self.Update_belief(Belief_N,Belief_P,0,ALPHA,BATE,GAMMA)


                Belief_N_previous = Belief_next_N
                Belief_P_previous = Belief_next_P
                done_0 = [0]
                done_1 = [1]
                
                if length == 2:
                    biospy = 1
                elif length == 3:
                    if row == 0:
                        biospy = length - 1
                    elif row != 0:
                        if Data_part.loc[row,'unique_ids'] == Data_part.loc[row-1,'unique_ids']:
                            biospy = 1
                        else:
                            biospy = 2

                
                Diagonsis1 = [0]
                if patient == 1:
                    reward1 = self.M_CT + self.loss_Geo(ob_state,index+1)
                elif patient == 0:
                    reward1 = self.B_CT

                '''
                if patient == 1:
                    Diagonsis = [2]
                    reward = self.M_M
                    invers_Diagonsis = [1]
                    invers_reward = self.M_B
                elif patient == 0:
                    Diagonsis = [1]
                    reward = self.B_B  
                    invers_Diagonsis = [2]
                    invers_reward = self.B_M
                '''

                self.Train_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(Diagonsis1), torch.tensor(reward1),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done_1), torch.tensor([start]),torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
            
                '''
                self.Train_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(Diagonsis), torch.tensor(reward),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done_0), torch.tensor([start]),torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
                
                self.Train_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(invers_Diagonsis), torch.tensor(invers_reward),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done_0), torch.tensor([start]),torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))   
                '''


    def initation_testing(self):

        Data_whole = pd.read_excel("All_Training_2022.xlsx")
        #ids_whole = Data_whole['unique_ids'].unique()
        #unique_ids_train = np.load('Training_selected_4-30_v2.npy')
        unique_ids_IN = np.load('Testing_selected_4-30_v2-1.npy')

        Data_part = Data_whole.loc[Data_whole['unique_ids'].isin(unique_ids_IN)]
        Data_part.index = range(len(Data_part))
        
        Belief_N_previous = -1
        Belief_P_previous = -1

        
        for row in range(len(Data_part)):
                    
            if Belief_N_previous < 0:
                #Belief_P = (Data_part.loc[row,'probability']/100).astype(np.float32) 
                Belief_P = (Data_part.loc[row,'value_N']).astype(np.float32) 
                Belief_N = ((1 - Belief_P)).astype(np.float32)
                #Belief_N = 0.01
                #Belief_N = Belief_P.astype(np.float32) 
                start = 1
                index = 0
            else:
                Belief_N = Belief_N_previous
                Belief_P = Belief_P_previous
                start = 0
                index = index + 1

            #print('aaa',Belief_N_previous)
            #print('bbb',Data_part.loc[row,'unique_ids'])
            #print('ccc',start)
            
            #ob_state = np.zeros(4).astype(np.float32)
            #ob_state[0] = Data_part.loc[row,'SCT_LONG_DIA']
            #ob_state[2] = Data_part.loc[row,'SCT_PRE_ATT_b']
            #ob_state[3] = Data_part.loc[row,'SCT_MARGINS_b'] 
            ob_state = np.zeros(1).astype(np.float32)
            ob_state[0] = Data_part.loc[row,'SCT_LONG_DIA'] 

            real_state = copy.deepcopy(ob_state)
            patient = Data_part.loc[row,'can_scr']
            traj_id = Data_part.loc[row,'unique_ids']
            trajectory = Data_part.loc[Data_part['unique_ids'] == traj_id]                
            length = len(trajectory)
            
            if row != len(Data_part)-1:            
                #ob_next_state = np.zeros(4).astype(np.float32)
                #ob_next_state[0] = Data_part.loc[row+1,'SCT_LONG_DIA']
                #ob_next_state[2] = Data_part.loc[row+1,'SCT_PRE_ATT_b']
                #ob_next_state[3] = Data_part.loc[row+1,'SCT_MARGINS_b']
                ob_next_state = np.zeros(1).astype(np.float32)
                ob_next_state[0] = Data_part.loc[row+1,'SCT_LONG_DIA']              
                real_next_state = copy.deepcopy(ob_next_state)
            else:
                ob_next_state = copy.deepcopy(ob_state)
                real_next_state = copy.deepcopy(real_state)
            
            
            #####  A  ###################################
            if row == len(Data_part)-1 or Data_part.loc[row,'unique_ids'] != Data_part.loc[row+1,'unique_ids']:

                Belief_N_previous = -1
                Belief_P_previous = -1
            
                done = [0]
                biospy = [0]
                                   
                if patient == 1:
                    Diagonsis = [2]
                    invers_Diagonsis = [1]
                elif patient == 0:
                    Diagonsis = [1]
                    invers_Diagonsis = [2]
                
                reward = [0]                 ################################
                invers_reward = [-20]        ################################      
                
                ob_next_state = copy.deepcopy(ob_state)
                real_next_state = copy.deepcopy(real_state)
                Belief_next_N = copy.deepcopy(Belief_N)
                Belief_next_P = copy.deepcopy(Belief_P)



                self.Test_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(Diagonsis), torch.tensor(reward),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done), torch.tensor([start]),torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
            
                #self.Test_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([ob_state]), torch.tensor([real_state]), torch.tensor(invers_Diagonsis), torch.tensor(invers_reward),                
                #                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), torch.tensor([real_next_state]), 
                #                            torch.tensor(done), torch.tensor(length), torch.tensor(biospy),torch.tensor(patient))  
                #                            
                #                                      
            elif Data_part.loc[row,'unique_ids'] == Data_part.loc[row+1,'unique_ids']:
                
                prob_1,prob_2 = self.Calculate_Prob_alpha(ob_state,ob_next_state)
                ALPHA = [prob_1,prob_2]
                prob_1,prob_2 = self.Calculate_Prob_beta(ob_state,ob_next_state)
                BATE = [prob_1,prob_2]
                prob_1,prob_2 = self.Calculate_Prob_gamma(ob_state,ob_next_state,1)
                GAMMA = [prob_1,prob_2]
                Belief_next_N, Belief_next_P = self.Update_belief(Belief_N,Belief_P,0,ALPHA,BATE,GAMMA)
                
                Belief_N_previous = Belief_next_N
                Belief_P_previous = Belief_next_P
                done = [1]
                
                if length == 2:
                    biospy = 1
                elif length == 3:
                    if row == 0:
                        biospy = length - 1
                    elif row != 0:
                        if Data_part.loc[row,'unique_ids'] == Data_part.loc[row-1,'unique_ids']:
                            biospy = 1
                        else:
                            biospy = 2

                Diagonsis1 = [0]
                reward1 = [-5]




                self.Test_Replay_memory.push(torch.tensor([Belief_N]), torch.tensor([Belief_P]), torch.tensor([index]), torch.tensor([ob_state]), torch.tensor(Diagonsis1), torch.tensor(reward1),                
                                            torch.tensor([Belief_next_N]), torch.tensor([Belief_next_P]),torch.tensor([ob_next_state]), 
                                            torch.tensor(done), torch.tensor([start]),torch.tensor(length), torch.tensor(biospy),torch.tensor(patient),torch.tensor(traj_id))  
            


