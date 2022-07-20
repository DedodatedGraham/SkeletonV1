# from random  import randint
# from sys import float_repr_style
import sys
# import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits import mplot3d
# import numpy as np
# import csv
# import scipy
# import pandas as pd
# import os
import time
# import pprofile
# import Skeletize
# import DataStructures 
# from DataStructures import kdTree
from SkeleNet import SkeleNet
from DataStructures import quicksort


sys.setrecursionlimit(10000)
st = time.time()

plt.rcParams['figure.dpi'] = 300

linep = True

#theta = np.linspace(0,2*np.pi,100)
tstart = time.time()
link = 'interface_points_070000.dat'
#link = 'spiral.dat'
#link = 'vof_points_norm_0650.dat'
#link = 't06.dat'
#link = 'bagdrop.dat'
if linep:
    link = '/home/graham_garcia1/SkeletonV1/' + link
net = SkeleNet(link)
net.solve(False)
net.plot([1,4,5])
net.savedat()

et = time.time()
tt = et - st

print('Total time to Complete: {} Minuites {} Seconds'.format(tt // 60, tt % 60))
