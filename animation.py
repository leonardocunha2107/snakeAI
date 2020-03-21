import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as AA
import numpy as np

x=np.linspace(0,2*np.pi,100)
frames=[]
for i in range(101):
    y=np.sin(x-i*2*np.pi/100)
    #fig,ax=plt.subplots()
    #ax.plot(x,y)
    #frames.append(fig)
    frames.append([plt.plot(x,y,animated=True)])
fig=plt.figure()
anim=AA(fig,frames)
anim.save('sin.gif')