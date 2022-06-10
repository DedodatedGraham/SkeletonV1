import os
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

import DataStructures
from DataStructures import kdTree

class SkeleNet:
    #In simpleTerms Skelenet is an easy to use skeletonization processer, 
    #It can intake a location of a data file, or even the straight points
    #Then itll process then and output different figures for different things
    #Can also Produce different shapes and stuff
    

    rnd = 0
###INITALIZERS
    def __init__(self, points,*,norms = []):
        self.IntPoints = []
        self.NormPoints = []
        #Determining type of points given
        if isinstance(points,str):    
            with open(points,'r') as csvfile:
                data = csv.reader(csvfile, delimiter = ' ')
                for row in data:
                    if len(row) == 2:#No Norms assoicated, Needs to find norms
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1])])
                    elif len(row) == 4:#Norms are given
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1])])
                            self.NormPoints.append([float(row[2]),float(row[3])]) 
                    elif len(row) == 3:#3D no norms    
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1]),floar(row[2])])
                    else:#3D with norm
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                            self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])]) 

            csvfile.close()
        elif isinstance(points,list):
            for point in points:
                self.IntPoints.append(point)
            if isinstance(norms,list):
                for norm in norms:
                    self.NormPoints.append(norm)
        

        
        if len(points[0]) == 3:
            self.dim = 3
        else:
            self.dim = 2
        #Gets normals if needed then normalizes the entire vectors
        if not(len(self.NormPoints) > 1):
            self.getNorms()

        if self.dim == 2:
            self.NormPoints = DataStructures.normalize2D(self.NormPoints)
        else:
            self.NormPoints = DataStructures.normalize3D(self.NormPoints)
        
#edit for plotting normals
        tx = []
        ty = []
        i = 0
        while i < len(self.IntPoints):
            tx.append(self.IntPoints[i][0])
            ty.append(self.IntPoints[i][1])
            i += 1
        i = 0
        while i < len(self.IntPoints):
            plt.xlim(-0.3,0.2)
            plt.ylim(-0.3,0.2)
            plt.scatter(tx,ty)
            plt.scatter(self.IntPoints[i][0],self.IntPoints[i][1]) 
            plt.plot([self.IntPoints[i][0] + self.NormPoints[i][0] * 1000,self.IntPoints[i][0] + self.NormPoints[i][0] * -1000],[self.IntPoints[i][1] + self.NormPoints[i][1] * 1000,self.IntPoints[i][1] + self.NormPoints[i][1] * -1000])
            save = os.getcwd() + "/Debug/Debug{:04d}.png".format(i)
            plt.savefig(save)
            plt.clf()
            i += 1
        #Determining other important 
        self.tag()




###MISC FUCNTIONS FOR SKELENET
    def tag(self):
        
        return

    def getNorms(self):
        tree = kdTree(self.IntPoints,self.dim)
        for point in self.IntPoints:
            if self.dim == 2:#2D Norms
                close1 = tree.getNearR(point,[])
                close2 = tree.getNearR(point,close1)
                if point == self.IntPoints[0]:
                    print(close1,close2)
                normA = [close1[1] - point[1],close1[0] - point[0]]
                normB = [close2[1] - point[1],close2[0] - point[0]]
                normP = [-1 * (normA[0] + normB[0]) / 2,-1 * (normA[1] + normB[1]) / 2]
                if point == self.IntPoints[0]:
                    print(normP)
                self.NormPoints.append(normP)
            else:#3D Norms
                return    

####ImageProcessing
