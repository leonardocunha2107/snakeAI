##Inspired by the work of Oscar Knagg, oscar@knagg.co.uk

from a2c_agents import TrajectoryStore, A2C, A2CModel,FancyModel
from game import SnakeGame
from itertools import count
import os
import torch
from torch.distributions import Categorical
import shutil
import numpy as np
import matplotlib.pyplot as plt
import

UPDATE_STEPS=10
GAMMA=0.999
SAVE_EVERY_EPS=100

class Logger:
    
    def __init__(self,name):
        keys=['loss','reward','snake_size']
        os.path.mkdir()
        self.store={k:[] for k in keys}
        self.steps_per_eps=[1]
        self.n_steps=0
        
    def moving_average(a, window_size=100) :
        ret = np.cumsum(np.array(a), dtype=float)
        ret[window_size:] = ret[window_size:] - ret[:-window_size]
        return ret[window_size - 1:] / window_size
    
    def push(self,done,**kwargs):
        ##To be called every step
        self.n_steps+=1
        for k,v in kwargs.items():
            if k in self.store:
                self.store[k].append(v)
        if done:
            self.steps_per_eps.append(1)
        else:
            self.steps_per_eps[-1]+=1
        if 'model' in kwargs:
            pass
        if 'env' in kwargs:
            self.store.snake_size.append()
            
    def per_episode(self,key):
        
    
          
    def save(self):
        pass

def train(num_episodes,name,board_shape=(5,5),lr=1e-4,**kwargs):
    
    save_dir=kwargs.get('save_dir','model/')
    optimizer=kwargs.get('optim',torch.optim.Adam)
    model=kwargs.get('model',FancyModel(num_actions=4, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                             num_residual_convs=2, num_feedforward=1, feedforward_dim=64))
    if save_dir:
        if os.path.exists(save_dir):
            print (f"Removing previous model at the folder {save_dir}")
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    env=SnakeGame(board_shape,device=device)
    a2c=A2C(model,GAMMA)
    trajectories = TrajectoryStore(device)
    optimizer=optimizer(model.parameters(),lr=lr)
    num_steps,num_eps=0,0
    last_step,total_reward=0,0
    state=env.get_board()
    
    for i_step in count(1):
        ## Acting on the env

        probs, state_value = model(state)
        action_distribution = Categorical(probs)
        entropy = action_distribution.entropy().mean()
        action = action_distribution.sample().clone().long()
        
        state, reward, done, _ = env.step(action)
        total_reward+=reward
        trajectories.append(
            action=action,
            log_prob=action_distribution.log_prob(action),
            value=state_value,
            reward=reward,
            done=done,
            entropy=entropy
        )
        
        loss=None
        if  i_step%UPDATE_STEPS==0:
            ## Compute losses and update model
            with torch.no_grad():
                _, bootstrap_values = model(state)
                
            value_loss, policy_loss = a2c.loss(bootstrap_values, trajectories.rewards, trajectories.values,
                                           trajectories.log_probs, trajectories.dones)
            ##Loss based on pi(s) entropy
            ##entropy_loss = - trajectories.entropies.mean()

            optimizer.zero_grad()
            loss = value_loss + policy_loss #+ entropy loss
            loss.requires_grad=True
            loss.backward()
            optimizer.step()

            trajectories.clear()
        
        #Logger.push(done,model=model,env=env,reward=reward,
         #           loss=loss.data if loss else None)
        if done: 
            num_eps+=1
            env.reset()
            last_step,total_reward=i_step,0
            
            if num_eps%SAVE_EVERY_EPS==0:
                torch.save(model.state_dict(),save_dir+f'{int(num_steps/SAVE_EVERY_EPS)}.mdl')
        if num_eps==num_episodes:
            print("Finished")
            break
        
            
        
if __name__=='__main__':
    train(2000)      
        

    