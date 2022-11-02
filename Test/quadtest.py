#Define Path for Folder
import sys
import os
source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/Src'
sys.path.insert(0,source)
sys.path.insert(0,source + r'/DataStruct')
from DataStructures import SplitTree
import numpy as np
import random as rand


xdat = []
ydat = []
rdat = []
indat = []

for i in range(3000):
    #Creates 300 pts along sinwave
    xdat.append((i*2*np.pi)/3000)
    ydat.append(np.cos(xdat[i]))
    rdat.append(i*0.01)

    #Adds Slight Randomness
    #if i % 10 == 0:
        #xdat[i] += rand.uniform(-0.5,0.5)
        #ydat[i] += rand.uniform(-0.5,0.5)
    #rdat[i] *= rand.uniform(-0.05,0.05)
    indat.append([xdat[i],ydat[i]])

a = SplitTree(indat,inrad = rdat)
a.purge(threshDistance = (2*np.pi/3000))
#a.Draw()

