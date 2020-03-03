##Inspired by the work of Oscar Knagg, oscar@knagg.co.uk


from itertools import count
import os
import torch
from torch.distributions import Categorical
import shutil
import numpy as np
import matplotlib.pyplot as plt
import json

def moving_average(a, window_size=200) :
        ret = np.cumsum(np.array(a), dtype=float)
        ret[window_size:] = ret[window_size:] - ret[:-window_size]
        return ret[window_size - 1:] / window_size

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
        
        rps_ax.set_title('Reward per Step (Moving Average)')
        rps_ax.plot(moving_average(self.rps))
        
        rpe_ax.set_title('Reward per episode')
        rpe_ax.plot(self.store['reward'])
        
        snake_ax.set_title('Snake size')
        snake_ax.plot(self.store['snake_size'])

        steps_ax.set_title('Steps per episode')
        steps_ax.plot(self.steps_per_eps)
        
        loss_ax.set_title('Loss')
        loss_ax.plot(self.store['loss'])
        
        self.fig.show()
        
    
          
    def save(self):
        with open(self.path+self.name+'.json','w+') as fd:
            json.dump(self.store,fd,indent=2)
        self.fig.savefig(self.path+self.name+'.png')
        """if self.colab:
            files.download(self.path+self.name+'.json')
            files.download(self.path+self.name+'.png')
        """
        
def train(num_episodes,name,board_shape=(5,5),lr=1e-4,colab=True,**kwargs):
    

    if colab:
        from .a2c_agents import TrajectoryStore, A2C, A2CModel,FancyModel
        from .game import SnakeGame
    else:
        from a2c_agents import TrajectoryStore, A2C,FancyModel
        from game import SnakeGame
        ##extract args
    
    on_noob=kwargs.get('on_noob','stay') ##what to do when the model tries to do opposite moves in succesion
    big_snake=kwargs.get('big_snake',True)  ##if the initial snake_length is 3, else is 1 
    mult_channels=kwargs.get('mult_channels',True)  ##If the representation we're feeding the model is on multiple channels or on only one
    in_channels=3 if mult_channels else 1
    #save_dir=kwargs.get('save_dir','model/')
    optimizer=kwargs.get('optim',torch.optim.AdamW)
    wall=kwargs.get('wall',False)  ##If the game has walls
    store_render=kwargs.get('store_render',False)
    plot_every=kwargs.get('plot_every',100)
    UPDATE_STEPS=kwargs.get('UPDATE_STEPS',20)
    GAMMA=kwargs.get('GAMMA',0.99)
    SAVE_EVERY_EPS=kwargs.get('SAVE_EVERY_EPS',1000)
    plot_every=kwargs.get('plot_every',100)
    path=kwargs.get('path','') ##Path to save model and logs

    if name == "fancy":
        model=kwargs.get('model',FancyModel(num_actions=4, num_initial_convs=2, in_channels=in_channels, conv_channels=32,
                             num_residual_convs=2, num_feedforward=1, feedforward_dim=64,colab=colab))
    elif name == "A2C":
        model=kwargs.get('model',A2CModel(in_channels=in_channels, n_actions=4, conv_channels=[32,32]))

    ##clear directory where we'll save our models
    """if save_dir:
        if os.path.exists(save_dir):
            print (f"Removing previous model at the folder {save_dir}")
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
    """
    ##Create Main objects
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    model=model.to(device)
    env=SnakeGame(board_shape,walls=wall,device=device,store_render=store_render,
                  on_noob=on_noob,big_snake=big_snake,mult_channels=mult_channels)
    a2c=A2C(model,GAMMA)
    trajectories = TrajectoryStore(device)
    optimizer=optimizer(model.parameters(),lr=lr)
    logger=Logger(name,path,plot_every,colab)
    
    num_eps=0
    state=env.get_board()
    
    for i_step in count(1):
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
        
        logger.push(done,model=model,env=env,reward=reward,
                   loss=float(loss) if loss else None)
        if done: 
            env.reset()
            num_eps+=1
            if num_eps%SAVE_EVERY_EPS==0 or num_eps==num_episodes:
                logger.save()
                torch.save(model.state_dict(),path+f'{int(num_eps/SAVE_EVERY_EPS)}.mdl')
                #if colab: files.download(path+f'{int(num_eps/SAVE_EVERY_EPS)}.mdl')
        if num_eps==num_episodes:
            print("Finished")
            break
        
            
        
if __name__=='__main__':
    train(50,'Local_Test',plot_every=5,colab=False)      
else:
    from google.colab import files

    

    