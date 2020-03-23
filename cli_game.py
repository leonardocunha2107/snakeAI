from game import SnakeGame
from os import system,name
import matplotlib.pyplot as plt
from array2gif import write_gif
import sys
from matplotlib.animation import ArtistAnimation as AA

def clear(): 
   ##Thanks stackoverflow
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 
        
if __name__=='__main__':
    game=SnakeGame((9,9),walls=True)
    clear()
    while (True):
        print(game.get_board())
        plt.imshow(game.render())
        plt.show()
        move=input("Move:\n")
        if move=='exit':
            sys.exit(0)
        move=int(move)
        
        if move in range(4):
            _,_,done,_=game.step(move)
        if done:
            #fig=plt.figure()
            #anim=AA(fig,[[plt.imshow(i)] for i in game.board_store])
            #anim.save('cli_game.gif')                        
            break
        print(done)
            
            