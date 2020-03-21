import numpy as np
from collections import deque
import gym
import random
import torch
import torch.nn.functional as F

SNAKE=1
FRUIT=2
BLACK     = ( 17.,  18.,  13.)
RED       = (255.,   0.,   0.)
GREEN     = (  0., 255.,   0.)
WHITE     = (255., 255., 255.)
directions=[(0,1),(-1,0),(0,-1),(1,0)]


def taxi_distance(t1,t2):
    s=0
    for  x,y in zip(t1,t2):
        s+=abs(x-y)
    return s
    
class SnakeGame(gym.Env):
    def __init__(self,dim=(10,10),walls=False,store_render=True,device='cuda',**kwargs):
        self.on_noob=kwargs.get('on_noob','stay')
        self.big_snake=kwargs.get('big_snake',True)
        self.mult_channels=kwargs.get('mult_channels',True)

        self.dim=dim
        self.action_space=gym.spaces.Discrete(4)
        self.observation_space=gym.spaces.Box(0,3,shape=self.dim)
        self.reward_range=(-1,1)
        self.device=device
        self.walls=walls
        self.observation_size=kwargs.get('observation_size',None)  ##tuple saying 
        assert not self.observation_size or \
            ( self.observation_size%2 ==0 and self.walls and self.observation_size<dim[0] and self.observation_size<dim[1])
        self.store_render=store_render
        self.reset()
        
        
    def get_board(self):
        if not self.mult_channels:
            return torch.tensor(self.board,device=self.device,dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        elif not self.observation_size:
            tb=torch.zeros(1,3,self.dim[0],self.dim[1],dtype=torch.float,device=self.device)
            tb[(0,0)+self.fruit]=1
            tb[(0,1)+self.snake[-1]]=1
            for tup in self.snake:
                tb[(0,2)+tup]=1
            return tb
        
        ##if we observe only the surroundings of the head
        observation_size=self.observation_size
        img = img.float() / 255
        
        # Pad envs so we ge tthe correct size observation even when the head of the snake
        # is close to the edge of the environment
        padding = [observation_size/2, observation_size/2, ] * 2
        padded_img = F.pad(img, padding)


        sh=self.snake[-1]
        observations = padded_img[
            sh[0]-observation_size/2:sh[0]+observation_size/2,sh[1]-observation_size/2:sh[1]+observation_size/2
        ]
        observations = observations.view(1,-1)

        return observations
        
        
        
        
    def reset(self):
        self.board=np.zeros(self.dim)
        self.snake=deque()
        self.empty=set([(i,j) for i in range(self.dim[0]) for j in range(self.dim[1])])
        
        snake_head=self.random_pos()        
        self.snake.append(snake_head)
        self.board[snake_head]=SNAKE
        
        if self.big_snake:   ##begin with snake of length 3
            dir1=random.randint(0,3)
            dir2=random.randint(0,3)
            if dir2!=dir1 and dir2%2==dir1%2:
                dir2=dir1
            snak=snake_head
            for d in [dir1,dir2]:
                snak=tuple(((snak[i]+directions[d][i]) for i in [0,1]))
                snak=tuple(snak[i]%self.dim[i] for i in [0,1])
                self.empty.remove(snak)
                self.snake.append(snak)
                self.board[snak]=SNAKE

        self.fruit=self.random_pos()
        self.board[self.fruit]=FRUIT
        
        self.t=0
        self.last_t_eat=0
        self.d=taxi_distance(snake_head,self.fruit)
        self.last_move=None
        if self.store_render: self.board_store=[self.render()]
        return self.board
    
    def random_pos(self):
        if not self.empty: return None
        pos=random.sample(self.empty,1) 
        self.empty.remove(pos[0])
        return pos[0]

    def step(self,action):
        snake_head=self.snake[-1]
        self.t+=1
        snake_head=tuple(((snake_head[i]+directions[action][i]) for i in [0,1]))
        ##Some death conditions
        if not self.walls: snake_head=tuple(snake_head[i]%self.dim[i] for i in range(2))
        if self.walls and (not 0<=snake_head[0]<self.dim[0]  \
                           or not 0<=snake_head[1]<self.dim[1]):
            return self.get_board(),-0.25,True,{}
        if self.on_noob=='stay' and self.last_move and len(self.snake)>1  \
            and action%2==self.last_move%2  and action!=self.last_move:
            return self.get_board(),-0.2,False,{}
        self.last_move=action
        if  self.board[snake_head]==SNAKE:
            return self.get_board(),-0.25,True,{}
        
        self.snake.append(snake_head)
        self.board[snake_head]=SNAKE
        if snake_head==self.fruit:
            self.fruit=self.random_pos() 
            self.board[self.fruit]=FRUIT
            self.d=taxi_distance(snake_head,self.fruit)
            self.last_t_eat=self.t
            if self.store_render: self.board_store.append(self.render())
            return self.get_board(),1,len(self.snake)==self.dim[0]*self.dim[1],{}
        
        self.empty.remove(snake_head)
        snake_tail=self.snake.popleft()
        self.empty.add(snake_tail)
        self.board[snake_tail]=0
        if self.store_render: self.board_store.append(self.render())
        return self.get_board(),0,False,{}
    """
        renders the game.
        Creates a rgb array with the colors corresponding to the current state of the game
        
        @:parameter mode mode of render request
                         'rgb' : @return array of rgb
    """
    def render(self):
            #grid = np.zeros((self.dim[0],self.dim[1],3))
            grid = np.zeros((self.dim[0],self.dim[1]), dtype = (int,3))
            
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    if self.board[i,j] == SNAKE:
                        grid[i,j] = np.array(GREEN)
                    elif self.board[i,j] == FRUIT:
                        grid[i,j] = np.array(RED)
            
            #We use a different color to the snake's head:
            head = self.snake[-1]
            grid[head[0],head[1]] = np.array(WHITE)

            return grid

        
        

            
        