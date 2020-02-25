import numpy as np
from collections import deque
import gym
import random
import torch

BLACK     = ( 17.,  18.,  13.)
RED       = (255.,   0.,   0.)
GREEN     = (  0., 255.,   0.)
WHITE     = (255., 255., 255.)
#directions=[(1,0),(-1,0),(0,1),(0,-1)]
directions=[(0,1),(-1,0),(0,-1),(1,0)]
"""directions=['RIGHT','LEFT','UP','DOWN']
d2v={d:v for d,v in zip(directions,[(1,0),(-1,0),(0,1),(0,-1)])}
"""
def taxi_distance(t1,t2):
    s=0
    for  x,y in zip(t1,t2):
        s+=abs(x-y)
    return s

class SnakeGame(gym.Env):
    SNAKE=1
    FRUIT=2
    def __init__(self,dim=(10,10),device='cuda'):
        self.dim=dim
        self.action_space=gym.spaces.Discrete(4)
        self.observation_space=gym.spaces.Box(0,3,shape=self.dim)
        self.reward_range=(-1,1)
        self.reset()
        self.device=device

    def get_board(self):
        tb=torch.zeros(1,3,self.dim[0],self.dim[1],dtype=torch.float,device=self.device)
        tb[(0,0)+self.fruit]=1
        tb[(0,1)+self.snake[-1]]=1
        for tup in self.snake:
            tb[(0,2)+tup]=1
        return tb
        #return torch.tensor(self.board,device=self.device,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    def reset(self):
        self.board=np.zeros(self.dim)
        self.snake=deque()
        self.empty=set([(i,j) for i in range(self.dim[0]) for j in range(self.dim[1])])
        snake_head=self.random_pos()
        
        self.snake.append(snake_head)
        self.board[snake_head]=self.SNAKE
        
        self.fruit=self.random_pos()
        self.board[self.fruit]=self.FRUIT
        
        self.t=0
        self.last_t_eat=0
        self.d=taxi_distance(snake_head,self.fruit)
        self.last_move=None
        
        return self.board
    
    def random_pos(self):
        pos=random.sample(self.empty,1)
        self.empty.remove(pos[0])
        return pos[0]

    def step(self,action):
        snake_head=self.snake[-1]
        self.t+=1
        snake_head=tuple(((snake_head[i]+directions[action][i])%self.dim[i] for i in [0,1]))
        
        if self.last_move and action%4==self.last_move%4:
            return self.get_board(),-0.5,False,{}
        
        self.last_move=action
        if  self.board[snake_head]==self.SNAKE:
            return self.get_board(),-0.25,True,{}
        
        self.snake.append(snake_head)
        self.board[snake_head]=self.SNAKE
        if snake_head==self.fruit:
            self.fruit=self.random_pos()
            self.board[self.fruit]=self.FRUIT
            self.d=taxi_distance(snake_head,self.fruit)
            self.last_t_eat=self.t
            return self.get_board(),1,len(self.snake)==self.dim[0]*self.dim[1],{}
        
        snake_tail=self.snake.popleft()
        self.empty.add(snake_tail)
        self.board[snake_tail]=0
        #rwd=(self.t-self.last_t_eat-self.d)
        #rwd=-np.sqrt(rwd)/np.sqrt(self.d)/8 if rwd >0 else 0
        rwd=0
        return self.get_board(),rwd,False,{}
    """
        renders the game.
        Creates a rgb array with the colors corresponding to the current state of the game
        
        @:parameter mode mode of render request
                         'rgb' : @return array of rgb
    """
    def render(self, mode = 'rgb'):
        if mode =='rgb':
            #grid = np.zeros((self.dim[0],self.dim[1],3))
            grid = np.zeros((self.dim[0],self.dim[1]), dtype = (int,3))
            
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    if self.board[i,j] == self.SNAKE:
                        grid[i,j] = np.array(GREEN)
                    elif self.board[i,j] == self.FRUIT:
                        grid[i,j] = np.array(RED)
            
            #We use a different color to the snake's head:
            head = self.snake.pop()
            grid[head[0],head[1]] = np.array(WHITE)
            self.snake.append(head)

            return grid
        else:
            raise('error: function not implemented')
            return -1
        
        

            
        