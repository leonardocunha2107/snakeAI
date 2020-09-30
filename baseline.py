import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from a2c import A2C,TrajectoryStore
import matplotlib.pyplot as plt
from itertools import count
device='cuda' if torch.cuda.is_available() else 'cpu'
LR=1e-3
UPDATE_STEPS=10
NUM_STEPS=1e3
class SimpleModel(nn.Module):
    def __init__(self,obs_d,action_d,h=50):
        super(SimpleModel,self).__init__()
        self.base=nn.Sequential(nn.Linear(obs_d,h),nn.ReLU(),
                                nn.Linear(h,h),nn.ReLU(),
                                nn.Linear(h,h),nn.ReLU()
                                )
        self.action_head=nn.Sequential(nn.Linear(h,action_d),nn.Softmax())
        self.value_head=nn.Linear(h,1)
    def forward(self,x):
        x=self.base(x)
        return self.action_head(x),self.value_head(x)

model=SimpleModel(4, 2).to(device=device)
trajectories=TrajectoryStore(device)
optimizer=torch.optim.Adam(model.parameters(),LR)
env =gym.make('CartPole-v0')
a2c=A2C(model,0.99)
state=env.reset()
state=torch.tensor(state,device=device,dtype=torch.float)
num_eps=0
rewards=[]
for i_step in count(1):
   

    
    probs, state_value = model(state)
    action_distribution = Categorical(probs)
    entropy = action_distribution.entropy().mean()
    action = action_distribution.sample().clone().long()
    state, reward, done, _ = env.step(int(action))
    state=torch.tensor(state,device=device,dtype=torch.float)
    trajectories.append(
        action=action,
        log_prob=action_distribution.log_prob(action),
        value=state_value,
        reward=reward,
        done=done,
        entropy=entropy
    )
    
    rewards.append(float(reward))
    
    if  i_step%UPDATE_STEPS==0 or done:
        _, bootstrap_value = model(state)
        bootstrap_value=bootstrap_value*(0 if done else 0)
        value_loss, policy_loss = a2c.loss(bootstrap_value, trajectories.rewards, trajectories.values,
                                           trajectories.log_probs,trajectories.dones)
        loss=value_loss+policy_loss
        loss.backward()
    if done:
        env.reset()
        num_eps+=1
    if i_step>=NUM_STEPS:
        break
    
    plt.plot(rewards)
    plt.savefig('baseline')
    
            