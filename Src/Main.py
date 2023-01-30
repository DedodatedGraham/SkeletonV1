from random  import randint
# from sys import float_repr_style
import sys
import getopt
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

#Settings/Modes
#PP-Mode or paralell procedure mode has 3 main opperations as of now
#Mode 0 => No Parallel Calculation(Default)
#Mode 1 => Normal Parallel Calculation
#Mode 2 => MPI Calulation


argumentList = sys.argv[1:]
options = "i:o:m:p:"
link = r''
savefile = r''
mode = 0
noderequest = 0
source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/'
try:
    arguments,argumentList = getopt.getopt(argumentList,options)
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-i"):
            link = str(currentValue)
        if currentArgument in ("-o"):
            savefile = str(currentValue)
        if currentArgument in ("-m"):
            mode = int(currentValue)
        if currentArgument in ("-p"):
            noderequest = int(currentValue)
except getopt.error as err:
    print(str(err))

if __name__ == '__main__':
    # __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    sys.setrecursionlimit(10000)
    st = time.time()

    plt.rcParams['figure.dpi'] = 300
    if len(link) <= 1:
        link = 'interface_points_070000.dat'
        # link = 'spiral.dat'
        # link = 'vof_points_norm_0650.dat'
        # link = 't06.dat'
        # link = 'vof_points_norms.dat'
        # link  = 'infc_50.dat'
        # link = 'bagdrop.dat'
        # link = 'disk1.dat'
    if len(savefile) <= 1:
        savefile = 'SkeleSave.dat'
        psavefile = 'PrePurge.dat'
    else:
        psavefile = savefile[:-4] + 'PrePurge.dat'
    link = source + r'SkeleData/Input/' + link
    psavefile = source + r'SkeleData/Output/' + psavefile
    savefile = source + r'SkeleData/Output/' + savefile
    #recover = source + r'SkeleData/SAVE/BESTSAVE02.dat'

    net = SkeleNet(link)
    net.solve(False,mode,noderequest)
    net.savedat(1,psavefile)
    #net.purge()
    #net.savedat(1,savefile)
    net.plot([1])

    et = time.time()
    tt = et - st
    print('Total time to Complete: {} Minuites {} Seconds'.format(tt // 60, tt % 60))
