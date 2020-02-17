import torch
import numpy as np
from queue import Queue
import gym
import random

directions=['RIGHT','LEFT','UP','DOWN']
d2v={d:v for d,v in zip(directions,[(1,0),(-1,0),(0,1),(0,-1)])}

def __mod__(tup1,tup2):
    return (t1%t2 for t1,t2 in zip(tup1,tup2))

class SnakeGame(gym.Env):
    self.SNAKE=1
    self.FRUIT=2
    def __init__(self,dim=(10,10)):
        self.dim=dim
        self.action_space=gym.spaces.Discrete(4)
        self.observation_space=gym.spaces.Box(0,3,shape=self.dim)
        self.reward_range=(-1,dim[0]*dim[1])
        self.reset()
    def reset(self):
        self.board=np.zeros(*self.dim)
        self.snake=Queue()
        self.empty=set([(i,j) for i in range(self.dim[0]) for j in range(self.dim[1])])
        snake_head=self.random_pos()
        
        
    def random_pos(self):
        while True:
            x,y=random.randint(0,self.dim[0]),random.randint(0,self.dim[1])
            (x,y)=__mod__((x,y),self.dim)
            if (x,y) in self.empty:
                self.empty.pop((x,y))
                return (x,y)
                
            
        