import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation as AA
import numpy as np
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--file')
args=parser.parse_args()

ls=np.load(args.file,allow_pickle=True)

fig=plt.figure()
anim=AA(fig,[[plt.imshow(i)] for i in ls])
anim.save(args.file.split('.')[0]+'.gif')