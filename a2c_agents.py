##Inspired by the work of Oscar Knagg, oscar@knagg.co.uk

import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable
EPS = 1e-8

class A2CModel(nn.Module):
    def __init__(self,in_channels=1,n_actions=4,
                 conv_channels=[32,32],kernel_sizes=[2,3]):
        super(A2CModel,self).__init__()



        
        assert len(conv_channels)==len(kernel_sizes)
        convs=[nn.Conv2d(in_channels,conv_channels[0],kernel_sizes[0]),nn.ReLU()]
        for i in range(1,len(conv_channels)):
            convs.append(nn.Conv2d(conv_channels[i-1],conv_channels[i],
                                   kernel_sizes[i]))
            convs.append(nn.ReLU())
        self.convs=nn.Sequential(*convs)
        
        self.value_head=nn.Linear(conv_channels[-1],1)
        self.policy_head=nn.Linear(conv_channels[-1],n_actions)
    def forward(self,x):
        x= self.convs(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        
        values = self.value_head(x)
        action_probabilities = self.policy_head(x)
        return F.softmax(action_probabilities, dim=-1), values
    
class A2C(object):
    """Class that encapsulates the advantage actor-critic algorithm.
    """
    def __init__(self,
                 actor_critic: nn.Module,
                 gamma: float,
                 value_loss_fn: Callable = F.smooth_l1_loss,
                 normalise_returns: bool = True):
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.normalise_returns = normalise_returns
        self.value_loss_fn = value_loss_fn

    def update(self, trajectories, state, done):
        """Calculates A2C losses based"""
        with torch.no_grad():
            _, bootstrap_value = self.actor_critic(state)

        R = bootstrap_value * (~done).float()
        returns = []
        for t in trajectories[::-1]:
            R = t.reward + self.gamma * R * (~t.done).float()
            returns.insert(0, R)

        returns = torch.stack(returns)

        if self.normalise_returns:
            returns = (returns - returns.mean()) / (returns.std() + EPS)

        values = torch.stack([transition.value for transition in trajectories])
        value_loss = F.smooth_l1_loss(values, returns).mean()
        advantages = returns - values
        log_probs = torch.stack([transition.log_prob for transition in trajectories]).unsqueeze(-1)
        policy_loss = - (advantages.detach() * log_probs).mean()

        entropy_loss = - torch.stack([transition.entropy for transition in trajectories]).mean()

        return value_loss, policy_loss, entropy_loss

    def loss(self,
             bootstrap_values: torch.Tensor,
             rewards: torch.Tensor,
             values: torch.Tensor,
             log_probs: torch.Tensor,
             dones: torch.Tensor):
        # Only take whats absolutely necessary for A2C
        # Leave states behind
        # Leave entropy calculation to another piece of code

        # print('rewards', rewards.shape)
        # print('values', values.shape)
        # print('log_probs', log_probs.shape)
        # print('dones', dones.shape)

        R = bootstrap_values * (~dones[-1]).float()
        returns = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (~d).float()
            returns.insert(0, R)

        returns = torch.stack(returns)
        # print('returns', returns.shape)

        if self.normalise_returns:
            returns = (returns - returns.mean()) / (returns.std() + EPS)
            returns=returns.view(-1,1)

        value_loss = self.value_loss_fn(values, returns).mean()
        advantages = returns - values
        # print('advantages', advantages.shape)
        policy_loss = - (advantages.detach() * log_probs).mean()

        return value_loss, policy_loss

class FancyModel(nn.Module):
    """Implementation of baseline agent architecture from https://arxiv.org/pdf/1806.01830.pdf"""
    def __init__(self,
                 in_channels: int,
                 num_initial_convs: int,
                 num_residual_convs: int,
                 num_feedforward: int,
                 feedforward_dim: int,
                 num_actions: int,
                 conv_channels: int = 16,
                 num_heads: int = 1, 
                 colab: bool =False):
        super(FancyModel, self).__init__()
        if colab: from .modules import ConvBlock, feedforward_block
        else: from modules import ConvBlock, feedforward_block
        self.in_channels = in_channels
        self.num_initial_convs = num_initial_convs
        self.num_residual_convs = num_residual_convs
        self.num_feedforward = num_feedforward
        self.feedforward_dim = feedforward_dim
        self.conv_channels = conv_channels
        self.num_actions = num_actions
        self.num_heads = num_heads

        initial_convs = [ConvBlock(self.in_channels, self.conv_channels, residual=False), ]
        for _ in range(self.num_initial_convs - 1):
            initial_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=False))

        self.initial_conv_blocks = nn.Sequential(*initial_convs)

        residual_convs = [ConvBlock(self.conv_channels, self.conv_channels, residual=True), ]
        for _ in range(self.num_residual_convs - 1):
            residual_convs.append(ConvBlock(self.conv_channels, self.conv_channels, residual=True))

        self.residual_conv_blocks = nn.Sequential(*residual_convs)

        feedforwards = [feedforward_block(self.conv_channels, self.feedforward_dim), ]
        for _ in range(self.num_feedforward - 1):
            feedforwards.append(feedforward_block(self.feedforward_dim, self.feedforward_dim))

        self.feedforward = nn.Sequential(*feedforwards)

        self.value_head = nn.Linear(self.feedforward_dim, num_heads)
        self.policy_head = nn.Linear(self.feedforward_dim, self.num_actions * num_heads)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.initial_conv_blocks(x)
        x = self.residual_conv_blocks(x)
        x = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.feedforward(x)
        values = self.value_head(x)
        action_probabilities = self.policy_head(x)
        return F.softmax(action_probabilities, dim=-1), values  
    
class TrajectoryStore(object):
    """Stores list of transitions.
    Each property should return a tensor of shape (num_steps, num_envs, 1)
    """
    def __init__(self,device):
        self.device=device
        self.clear()

    def append(self,
               state: torch.Tensor = None,
               action: torch.Tensor = None,
               log_prob: torch.Tensor = None,
               reward: torch.Tensor = None,
               value: torch.Tensor = None,
               done: torch.Tensor = None,
               entropy: torch.Tensor = None):
        """Adds a transition to the store.
        Each argument should be a vector of shape (num_envs, 1)
        """
        if state is not None:
            self._states.append(state)

        if action is not None:
            self._actions.append(action)

        if log_prob is not None:
            self._log_probs.append(torch.tensor([log_prob],dtype=torch.float,device=self.device))

        if reward is not None:
            self._rewards.append(torch.tensor([reward],dtype=torch.float,device=self.device))

        if value is not None:
            self._values.append(torch.tensor([value],dtype=torch.float,device=self.device))

        if done is not None:
            done=1 if done else 0
            self._dones.append(torch.tensor([done],dtype=torch.int,device=self.device))

        if entropy is not None:
            self._entropies.append(entropy)

    def clear(self):
        self._states = []
        self._actions = []
        self._log_probs = []
        self._rewards = []
        self._values = []
        self._dones = []
        self._entropies = []

    @property
    def states(self):
        return torch.stack(self._states)

    @property
    def actions(self):
        return torch.stack(self._actions)

    @property
    def log_probs(self):
        return torch.stack(self._log_probs).unsqueeze(-1)

    @property
    def rewards(self):
        return torch.stack(self._rewards)

    @property
    def values(self):
        return torch.stack(self._values)

    @property
    def dones(self):
        return torch.stack(self._dones)

    @property
    def entropies(self):
        return torch.stack(self._entropies)