from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import numpy as np
import csv
import scipy
import pandas as pd
import os
import time

import Skeletize
import DataStructures 
from DataStructures import kdTree
from SkeleNet import SkeleNet

plt.rcParams['figure.dpi'] = 300

theta = np.linspace(0,2*np.pi,100)

xmin = -0.3
xmax = 0.2
ymin = -0.3
ymax = 0.2

tstart = time.time()
IntPoints = []
TestPoints = []
NormPoints = []


net = SkeleNet('interface_points_070000.dat')
#net = SkeleNet('spiral.dat')
#net = SkeleNet('vof_points_norm.dat')
net.solve(True)
net.plot([1])



