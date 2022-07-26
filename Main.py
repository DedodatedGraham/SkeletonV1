from random  import randint
# from sys import float_repr_style
import sys
# import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits import mplot3d
# import numpy as np
import csv
# import scipy
# import pandas as pd
import os
import time
# import pprofile
# import Skeletize
# import DataStructures 
# from DataStructures import kdTree
from SkeleNet import SkeleNet
from DataStructures import kdTree
#theta = np.linspace(0,2*np.pi,100)



# if __name__ == '__main__':
#     __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
sys.setrecursionlimit(10000)
#     st = time.time()
    
plt.rcParams['figure.dpi'] = 300
    
linep = True
    
    
    # link = r'\interface_points_070000.dat'
    # link = r'\spiral.dat'
link = r'\vof_points_norm_0650.dat'
    # link = r'\t06.dat'
    # link = r'\bagdrop.dat'
if linep:
    source = os.getcwd()
    link = source + link
#     net = SkeleNet(link)
#     net.solve(False)
#     net.savedat(1)
#     net.plot([1])
    
#     et = time.time()
#     tt = et - st
    
#     print('Total time to Complete: {} Minuites {} Seconds'.format(tt // 60, tt % 60))


    #k-d Tree Speed Test
print('opening')
IntPoints = []
NormPoints = []
j = 0
with open(link,'r') as csvfile:
    data = csv.reader(csvfile, delimiter = ' ')
    for row in data:
        j += 1
        print(j)
        size = len(row)
        if str(row[0]) == 'x':#if title
            a = 1    
        else:
            IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
            NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
tree = kdTree(IntPoints)
i = 0
while i < 100:
    print(kdTree.getNearR(IntPoints[randint(0,len(IntPoints) - 1)] + 0.01 * randint(0,100),[]))
    i += 1
    
                