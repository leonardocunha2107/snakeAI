from game import SnakeGame
from os import system,name
import matplotlib.pyplot as plt

def clear(): 
   ##Thanks stackoverflow
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 
        
if __name__=='__main__':
    game=SnakeGame()
    while (True):
        plt.imshow(game.render())
        plt.show()
        move=input("Move:\n")
        if move=='exit':
            sys.exit(0)
        move=int(move)
        if move in range(4):
            _,_,done,_=game.step(move)
        print(done)
            
            