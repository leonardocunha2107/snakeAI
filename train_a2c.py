##Inspired by the work of Oscar Knagg, oscar@knagg.co.uk


from itertools import count
import os
import torch
from torch.distributions import Categorical
import torch.nn as nn
import shutil
import numpy as np
from matplotlib.animation import ArtistAnimation as AA
import matplotlib.pyplot as plt
import json
import random
import traceback
from time import time

MAX_GRAD_NORM = 0.5


def moving_average(a, window_size=500) :
        ret = np.cumsum(np.array(a), dtype=float)
        ret[window_size:] = ret[window_size:] - ret[:-window_size]
        return ret[window_size - 1:] / window_size
def cum_average(a):
    return np.cumsum(a)/np.arange(1,len(a)+1)

class Logger:
    
    def __init__(self,name,path='',plot_every=100,colab=True):
        self.colab=colab
        self.name=name
        self.plot_every=plot_every
        self.path=path
        self.keys=['loss','reward','snake_size']
        self.store={k:[] for k in self.keys}
        self.temp={k:[] for k in self.keys}
        self.rps=[]  ##reward per step
        self.steps_per_eps=[1]
        self.n_steps=0
        self.max_reward=0
        self.best_game=None
        self.num_eps=0
        
        if colab:
            from IPython import display
            self.display=display
        self.fig=plt.figure(figsize=(9.,8.))
        
    
    def push(self,done,**kwargs):
        ##To be called every step
        self.n_steps+=1
        for k,v in kwargs.items():
            if k in self.temp:
                self.temp[k].append(v)
        if 'env' in kwargs:
            self.temp['snake_size'].append(len(kwargs['env'].snake))
        if 'model' in kwargs:
            pass  ##TODO
        
        ## If the episode we were logging is over
        if done:
            aux=[t for t in self.temp['loss'] if t]
            self.store['loss'].append(sum(aux)/len(aux) if aux else 0)
            self.store['snake_size'].append(self.temp['snake_size'][-1])
            aux=self.temp['reward']
            self.rps.extend(aux)
            self.store['reward'].append(sum(aux)/len(aux) if aux else 0)
            if self.store['reward'][-1]>self.max_reward:
                self.max_reward=self.store['reward'][-1]
                if 'env' in kwargs and kwargs['env'].store_render:
                    self.best_game=kwargs['env'].board_store
            self.steps_per_eps.append(1)
            self.temp={k:[] for k in self.keys}
            self.num_eps+=1
            if self.num_eps%self.plot_every==0:
                print(f'Done with eps {self.num_eps}')
                self.plot()


            
        else: self.steps_per_eps[-1]+=1
        
        
                    
    def plot(self):
        self.fig.clear()
        ((rps_ax,rpe_ax,snake_ax),(steps_ax,loss_ax,_))=self.fig.subplots(2,3)
        
        rps_ax.set_title('Reward per Step (MAVG)')
        rps_ax.plot(moving_average(self.rps))
        
        rpe_ax.set_title('Reward per episode (MAVG)')
        rpe_ax.plot(moving_average(self.store['reward']))
        
        snake_ax.set_title('Snake size (MAVG)')
        snake_ax.plot(moving_average(self.store['snake_size']))

        steps_ax.set_title('Steps per episode')
        steps_ax.plot(self.steps_per_eps)
        
        loss_ax.set_title('Loss (MAVG)')
        loss_ax.plot(moving_average(self.store['loss']))
        
        if self.colab:    
            self.display.clear_output(wait=True)
            self.display.display(self.fig)
        else:
            self.fig.show()
            
        
    
          
    def save(self):
        with open(self.path+self.name+'.json','w+') as fd:
            json.dump(self.store,fd,indent=2)
        self.fig.savefig(self.path+self.name+'.png')
        
        np.save(self.path+self.name+'.npy',self.best_game)      
        """if self.colab:
            files.download(self.path+self.name+'.json')
            files.download(self.path+self.name+'.png')
        """
        
def train(num_steps,name,board_shape=(9,9),lr=1e-4,colab=True,**kwargs):
    

    if colab:
        from .a2c_agents import TrajectoryStore, A2C,A2CModel,FancyModel,FeedforwardModel
        from .game import SnakeGame
    else:
        from a2c_agents import TrajectoryStore, A2C,A2CModel,FancyModel,FeedforwardModel
        from game import SnakeGame
        ##extract args
    
    on_noob=kwargs.get('on_noob','stay') ##what to do when the model tries to do opposite moves in succesion
    big_snake=kwargs.get('big_snake',True)  ##if the initial snake_length is 3, else is 1 
    mult_channels=kwargs.get('mult_channels',True)  ##If the representation we're feeding the model is on multiple channels or on only one
    in_channels=3 if mult_channels else 1
    #save_dir=kwargs.get('save_dir','model/')
    optimizer=kwargs.get('optim',torch.optim.RMSprop)
    wall=kwargs.get('wall',False)  ##If the game has walls
    store_render=kwargs.get('store_render',True)
    observation_size=kwargs.get("observation_size",None)
    plot_every=kwargs.get('plot_every',100)
    UPDATE_STEPS=kwargs.get('UPDATE_STEPS',40)
    GAMMA=kwargs.get('GAMMA',0.99)
    SAVE_EVERY_EPS=kwargs.get('SAVE_EVERY_EPS',1000)
    plot_every=kwargs.get('plot_every',100)
    path=kwargs.get('path','') ##Path to save model and logs
    normalise_returns=kwargs.get('normalise_returns',False)
    
    seed=kwargs.get('seed',50)   ##seeding
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    random.seed(seed)

    model=kwargs.get('model',"fancy")
    
    if type(model)==str:           ##choose model
        if observation_size:
            model=FeedforwardModel(4,2,64,\
                    num_inputs=in_channels*(observation_size**2))
        elif model == "fancy":
            model=FancyModel(num_actions=4, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                                 num_residual_convs=2, num_feedforward=1, feedforward_dim=64,colab=colab)
        elif model == "A2C":
            model=A2CModel(in_channels=in_channels, n_actions=4, conv_channels=[32,32])


    ##Create Main objects
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    env=SnakeGame(board_shape,walls=wall,device=device,store_render=store_render,
                  on_noob=on_noob,big_snake=big_snake,mult_channels=mult_channels,observation_size=observation_size)
    a2c=A2C(model,GAMMA,normalise_returns=normalise_returns)
    trajectories = TrajectoryStore(device)
    optimizer=optimizer(model.parameters(),lr=lr)
    logger=Logger(name,path,plot_every,colab)
    
    num_eps=0
    state=env.get_board()
    
    for i_step in count(1):
        try:
            ## Acting on the env
            probs, state_value = model(state)
            action_distribution = Categorical(probs)
            entropy = action_distribution.entropy().mean()
            action = action_distribution.sample().clone().long()
            state, reward, done, _ = env.step(action)

            trajectories.append(
                action=action,
                log_prob=action_distribution.log_prob(action),
                value=state_value,
                reward=reward,
                done=done,
                entropy=entropy
            )
            loss=None
            if  i_step%UPDATE_STEPS==0 or done:
                ## Compute losses and update model
                with torch.no_grad():
                    _, bootstrap_value = model(state)
                    bootstrap_value=bootstrap_value*(0 if done else 0)
                    
                value_loss, policy_loss = a2c.loss(bootstrap_value, trajectories.rewards, trajectories.values,
                                               trajectories.log_probs)
                ##Loss based on pi(s) entropy
                ##entropy_loss = - trajectories.entropies.mean()
    
                optimizer.zero_grad()
                loss = value_loss + policy_loss #+ entropy loss
                loss.requires_grad=True
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                optimizer.step()
    
                trajectories.clear()


            logger.push(done,model=model,env=env,reward=reward,
                       loss=float(loss) if loss else None)
            if done:
                env.reset()
                num_eps+=1
                if num_eps%SAVE_EVERY_EPS==0 or i_step>=num_steps:
                    logger.save()
                    torch.save(model.state_dict(),path+name+f'{int(num_eps/SAVE_EVERY_EPS)}.mdl')
            if i_step>=num_steps:
                print(f"Finished {name}")
                break
        except:
            print(traceback.format_exc())

            return logger
    return logger
        
            
        
if __name__=='__main__':    
    try: 
        train(1e8,'conv_walls',lr=5e-4 ,colab=False,plot_every=10000,
              board_shape=(9,9),path='experiments/',model='fancy',SAVE_EVERY_EPS=20000)
    except: 
        print("fail 1")
        print(traceback.format_exc())
    
    try: 
        train(1e7,'conv_walls_no_normalise',lr=5e-4 ,colab=False,plot_every=10000,
              board_shape=(9,9),path='experiments/',model='fancy',wall=True,normalise_returns=False,SAVE_EVERY_EPS=20000)
    except: 
        print("fail 2")
        print(traceback.format_exc())

    try: 
        train(1e7,'local_view',lr=1e-3 ,colab=False,plot_every=10000,
              board_shape=(9,9),path='experiments/',model='fancy',observation_size=2,wall=True,SAVE_EVERY_EPS=20000)
    except: 
        print("fail 3")
        print(traceback.format_exc())

    

    