import os
from random  import randint
import copy
import sys
from sys import float_repr_style
# import matplotlib
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import numpy as np
import csv
# import scipy
import pandas as pd
import time
import multiprocessing as mp
from pathos.pp import ParallelPool
from pathos.multiprocessing import ProcessingPool,ThreadPool

from DataStructures import kdTree,SplitTree
from Skeletize import checkRepeat,getRadius,getDistance,normalize, getAngle, getPoint,randPN
from mpl_toolkits import mplot3d
from itertools import cycle
source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/'
infile = source + r'SkeleData/Output/bench.txt'
with open(infile,'r') as csvfile:
    data = csv.reader(csvfile)
    time = []
    nodes = []
    i = 0
    for row in data:
        if i == 0:
            a = 1
        else:
            time.append(float(row[0]))
            nodes.append(pow(2,i-1))
        i += 1
i = 0
while i < len(time):
    time[i] = np.log(time[i])
    nodes[i] = np.log(nodes[i])
    i += 1
print(time,nodes)

plt.plot(nodes,time)
plt.xlabel('Log(Cores); (1->128)')
plt.ylabel('Log(Time)')
plt.savefig('LogBench.png')
