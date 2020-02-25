##Inspired by the work of Oscar Knagg, oscar@knagg.co.uk

from .a2c import TrajectoryStore, A2C, A2CModel
from .game import SnakeGame
from itertools import count
import os
import torch
from torch.distributions import Categorical

BOARD_SHAPE=(5,5)
UPDATE_STEPS=10
GAMMA=0.999
SAVE_EVERY_EPS=100

def train(num_episodes,save_dir='model/',lr=1e-3):
    if save_dir:
        if os.path.exists(save_dir):
            print (f"Removing previous model at the folder {save_dir}")
            os.rmdir(save_dir)
        os.mkdir(save_dir)
        
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    env=SnakeGame(BOARD_SHAPE,device=device)
    model=A2CModel(BOARD_SHAPE,in_channels=3).to(device)
    a2c=A2C(model,GAMMA)
    trajectories = TrajectoryStore(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
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
        
        if done: 
            num_eps+=1
            s=f'Done with eps {num_eps} with reward {total_reward} and '
            s+=f'{i_step-last_step} steps'
            print(s)
            env.reset()
            last_step,total_reward=i_step,0
            
            if num_eps%SAVE_EVERY_EPS==0:
                torch.save(model.state_dict(),save_dir+f'{int(num_steps/SAVE_EVERY_EPS)}.mdl')
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
            
        
if __name__=='__main__':
    train(10)      
        

    