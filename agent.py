from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import math
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 2000
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
device=torch.cuda.device(0)
class SimpleDQN(nn.Module):
    def __init__(self,board_shape,n_actions=4,h=100):
        super(SimpleDQN,self).__init__()
        self.n_actions=n_actions
        #self.flatter=nn.Flatten(start_dim=0)
        self.fc1=nn.Linear(board_shape[0]*board_shape[1],h)
        self.fc2=nn.Linear(h,n_actions)
    def forward(self,x):
        B=None
        if len(x.shape)>2:
            B=x.shape[0]
            x=x.view(B,-1)
            
        else:
            B=1
            x=x.view(1,-1)
        x=F.relu(self.fc1(x))
        return self.fc2(x).view(B,self.n_actions)
    
    
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.h,self.w=h,w
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 2, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        if len(x.shape)==2:
            x=x.view(1,1,self.h,self.w)
        else:
            x=x.view(x.shape[0],1,self.h,self.w)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self,board_shape,model='cnn',device='cuda',n_actions=4):
        self.batch_size=128
        self.gamma=0.999
        self.n_actions=n_actions
        self.device=device
        if model=='cnn':
            self.net =DQN(board_shape[0],board_shape[1],n_actions).to(device)
        else:
            self.net=SimpleDQN(board_shape,n_actions=n_actions).to(device)
        self.steps_done=0
    def select_action(self,state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        
        