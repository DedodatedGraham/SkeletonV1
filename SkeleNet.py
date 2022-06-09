
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
    #It can intake a location of a data file, or even the straight points
    #Then itll process then and output different figures for different things
    #Can also Produce different shapes and stuff
    

    rnd = 0
###INITALIZERS
    def __init__(self, FileLocation : str):
        self.IntPoints = []
        self.TestPoints = []
        self.NormPoints = []
        with open(FileLocation,'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ' ')
            for row in data:
                self.IntPoints.append([float(row[0]),float(row[1])])
                if len(row) == 2:#No Norms assoicated, Needs to find norms
                    if randint(0,10) >= self.rnd:
                        self.TestPoints.append([float(row[0]),float(row[1])])
                        self.dimensions = 2
                elif len(row) == 4:#Norms are given
                    if randint(0,10) >= self.rnd:
                        self.TestPoints.append([float(row[0]),float(row[1])])
                        self.NormPoints.append([float(row[2]),float(row[3])]) 
                        self.dimensions = 2
                elif len(row) == 3:#3D no norms    
                    if randint(0,10) >= self.rnd:
                        self.TestPoints.append([float(row[0]),float(row[1]),floar(row[2])])
                        self.dimensions = 3
                else:#3D with norm
                    if randint(0,10) >= self.rnd:
                        self.TestPoints.append([float(row[0]),float(row[1]),float(row[2])])
                        self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])]) 
                        self.dimensions = 3

        csvfile.close()

    def __init__(self,points : list):
        self.IntPoints = []
        self.TestPoints = []
        self.NormPoints = []
        for point in points:
            if randint(0,10) >= self.rnd:
                self.TestPoints.append(point)
        if len(points[0])==2:
            self.dimensions = 2
        else:
            self.dimensions = 3 
        self.NormPoints = getNorms(self.TestPoints)

    def __init__(self,points : list, norms : list):
        self.IntPoints = []
        self.TestPoints = []
        self.NormPoints = []
        for point in points:
            if randint(0,10) >= self.rnd:
                self.TestPoints.append(point)	
        for norm in norms:
            if randint(0,10) >= self.rnd:
                self.NormPoints.append(norm)
        if len(points[0])==2:
            self.dimensions = 2
        else:
            self.dimensions = 3 




####ImageProcessing

###MISC FUCNTIONS FOR SKELENET
    def getNorms(points : list) -> list:
    #Used for determining norms if the given data doesnt have any
        return []
