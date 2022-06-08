
from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import numpy as np
import csv
import scipy
import pandas as pd



class SkeleNet:
	#In simpleTerms Skelenet is an easy to use skeletonization processer, 
	#as before this theres only a messy Main file and a few functions, this aims
	#to tie both of these things together well and mangage itsself well.

	def __init__(self, FileLocation : str):
		rnd = 0
		with open('interface_points_070000.dat','r') as csvfile:
			data = csv.reader(csvfile, delimiter = ' ')
        		for row in data:
            		IntPoints.append([float(row[0]),float(row[1])])
				if len(row) == 2:#No Norms assoicated, Needs to find norms
					if randint(0,10) >= rnd:
                		TestPoints.append([float(row[0]),float(row[1])])
        		else:#Norms are given
					if randint(0,10) >= rnd:
                		TestPoints.append([float(row[0]),float(row[1])])
               			NormPoints.append([float(row[2]),float(row[3])]) 
			csvfile.close()

	def __init__(self,pts : List, nrms : List):
		self.points = []
		self.norms = [] 
		self.points.append(for point in pts)
		self.norms.append(for norm in nrms)

	
	def __init__(self, pts : List):
		self.points = []
		for point in pts:
		self.points.append(points) 
		
		self.norms = DataStructures.findNorm(pts)


	
