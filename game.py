import torch
import numpy as np
from collections import deque
import gym
import random

directions=[(1,0),(-1,0),(0,1),(0,-1)]
"""directions=['RIGHT','LEFT','UP','DOWN']
d2v={d:v for d,v in zip(directions,[(1,0),(-1,0),(0,1),(0,-1)])}
"""
def __mod__(tup1,tup2): ##find actual position of snake in the board
    return (t1%t2 for t1,t2 in zip(tup1,tup2))

class SnakeGame(gym.Env):
    self.SNAKE=1
    self.FRUIT=2
    def __init__(self,dim=(10,10)):
        self.dim=dim
        self.action_space=gym.spaces.Discrete(4)
        self.observation_space=gym.spaces.Box(0,3,shape=self.dim)
        self.reward_range=(-1,1)
        self.reset()
        
    def reset(self):
        self.board=np.zeros(self.dim)
        self.snake=deque()
        self.empty=set([(i,j) for i in range(self.dim[0]) for j in range(self.dim[1])])
        snake_head=self.random_pos()
        
        self.snake.append(snake_head)
        self.board[snake_head]=self.SNAKE
        
        self.fruit=self.random_pos()
        self.board[self.fruit]=self.FRUIT
    
    def random_pos(self):
        pos=random.sample(self.empty,1)
        self.empty.pop(pos)
        return pos

    def step(self,action):
        snake_head=self.snake.pop()
        self.snake.append(snake_head)
        snake_head=((snake_head[i]+directions[action][i])%self.dim[i] for i in [0,1])
        
        if  snake_head!=self.fruit and snake_head not in self.empty:
            return self.board,-1.,True,{}
        
        self.snake.append(snake_head)
        self.board[snake_head]=self.SNAKE
        if snake_head==self.fruit:
            self.
        
        
    
        
        
   
                
            
        