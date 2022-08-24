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
from SkeleNet import SkeleNet,skeletize
from Skeletize import normalize, getDistance
from DataStructures import kdTree
#theta = np.linspace(0,2*np.pi,100)



if __name__ == '__main__':
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    sys.setrecursionlimit(10000)
    st = time.time()
    
    plt.rcParams['figure.dpi'] = 300

    linep = True
    
    
    # link = r'/interface_points_070000.dat'
    # link = r'/spiral.dat' 
    # link = r'/vof_points_norm_0650.dat'
    # link = r'/t06.dat'
    link = r'/vof_points_norms.dat'
    # link = r'/bagdrop.dat'
    # link = r'/disk1.dat'
    if linep:
        source = os.path.dirname(os.path.abspath(__file__))
        link = source + link
    
    net = SkeleNet(link)
    net.solve(True)
    net.savedat(1)
    net.plot([1,2])
    
    et = time.time()
    tt = et - st
    print('Total time to Complete: {} Minuites {} Seconds'.format(tt // 60, tt % 60))


    #k-d Tree Speed Test
# print('opening')
# IntPoints = []
# NormPoints = []
# j = 0
# with open(link,'r') as csvfile:
#     data = csv.reader(csvfile, delimiter = ' ')
#     for row in data:
#         j += 1
#         print(j)
#         size = len(row)
#         if str(row[0]) == 'x':#if title
#             a = 1    
#         else:
#             if len(row) > 5:
#                 IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
#                 NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
#             else:
#                 IntPoints.append([float(row[0]),float(row[1])])
#                 NormPoints.append([float(row[2]),float(row[3])])
# tree = kdTree(IntPoints)
# norms = normalize(NormPoints)
# tot = 0
# tpt = []
# j = 0
# while j < min(20,len(IntPoints)):
#     tpt = IntPoints[randint(0, len(IntPoints) - 1)]
#     tts = time.time()
#     tot += getDistance(tpt,tree.getNearR([tpt,[],False]))
#     tte = time.time()
#     print('thresh',j,'took {} secs'.format((tte - tts) % 60))
#     j += 1
# threshDistance = tot / min(20,len(IntPoints)) 
# pts = []
# nrms = []
# i = 0
# while i < len(IntPoints):
#     print(i)
#     pts.append(IntPoints[i])
#     nrms.append(norms[i])
#     i += randint(20,10000)
#     i += 1
#     # i += randint(1,10)
# out = skeletize(pts,nrms, threshDistance, tree)
    
                
