#Define Path for Folder
import sys
import os
source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/Src'
sys.path.insert(0,source)
sys.path.insert(0,source + r'/DataStruct')
from DataStructures import SplitTree
import numpy as np
import random as rand
import matplotlib.pyplot as plt

xdat = []
ydat = []
rdat = []
indat = []

for i in range(4000):
    #Creates 300 pts along sinwave
    if i < 3000:
        xdat.append((i*2*np.pi)/3000)
        ydat.append(np.cos(xdat[i]))
        rdat.append(i*0.01)
    else:
        xdat.append(((i-3000)/1000) + 1 )
        ydat.append(xdat[i]*((i-3000)/1000))
        rdat.append(i*0.01)

    #Adds Slight Randomness
    if i % 10 == 0:
        xdat[i] += rand.uniform(-0.5,0.5) 
        ydat[i] += rand.uniform(-0.5,0.5)     
        #rdat[i] *= rand.uniform(-0.05,0.05)
    indat.append([xdat[i],ydat[i]])

a = SplitTree(indat,inrad = rdat)
plt.scatter(xdat,ydat,s=5)
save = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/Plot/prequad.png'
plt.savefig(save)
plt.clf()
a.purge(threshDistance = (2*np.pi/3000))
#a.Draw()

