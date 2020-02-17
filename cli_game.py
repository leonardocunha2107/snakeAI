from game import SnakeGame
from os import system,name

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
        print("{}\n\n".format(game))
        move=input("Move:\n")
        if move=='exit':
            sys.exit(0)
        move=int(move)
        if move in range(4):
            _,_,done,_=game.step(action)
            
            