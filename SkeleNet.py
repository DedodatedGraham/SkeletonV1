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
import time

from DataStructures import kdTree,getDistance2D,getDistance3D,normalize2D,normalize3D
from Skeletize import checkRepeat,getRadius2D,getRadius3D

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
        self.MasterTag = []
        self.tagKey = []
        self.tpoints = []
        self.tnorms = []
        #Determining type of points given
        if isinstance(points,str):    
            with open(points,'r') as csvfile:
                data = csv.reader(csvfile, delimiter = ' ')
                for row in data:
                    size = len(row)
                    if size == 4:#2D w/ no tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1])])
                            self.NormPoints.append([float(row[2]),float(row[3])]) 
                            self.MasterTag.append(randint(0,3))
                    elif size == 5:#2D w/ tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1])])
                            self.NormPoints.append([float(row[2]),float(row[3])]) 
                            self.MasterTag.append(int(row[4]))
                    elif size == 6:#3D w/ no tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                            self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
                    elif size == 7:#3D w/ tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                            self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
                            self.MasterTag.append(int(row[6]))
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
            self.getNorms()#Caution, normals only acurate to locals and perfect shapes and such, use for Smooth Points
            
        if self.dim == 2:
            self.NormPoints = normalize2D(self.NormPoints)
        else:
            self.NormPoints = normalize3D(self.NormPoints)
        
        self.tag()

###MAIN FUNCTIONS
    def skeletize(self,key : int,animate : bool):
        #Skeletize takes in 
        #FROM INPUT
        #key is the specific index of tpoints and tnorms, allows for
        #parallel capabilities in splitting apart skeleton tasks 
        #FROM CLASS
        #points, the given plain list of points [x,y] for 2D case
        #norms, a list of not yet normalized normal points [n_x,n_y] here for 2D case
        #then returns 2 things
        # finPoints = [[x1,y1],...] of skeleton points
        # finR = [r1,...] of the radius of each skeleton point
        
        self.threshDistance = []
        print('Skeletizing #{}...'.format(key))
        pts = []
        
        ##INITAL SETTING UP METHOD
        #removes the connection between input points and output ponts
        #When not doing this kd-tree skews points for some reason?
        i = 0
        while i < len(self.tpoints[key]):
            pts.append(self.tpoints[key][i])
            i += 1
        tree = kdTree(pts,self.dim)        
        #Threshdistance averages 3 random points distance incase there is 
        #An adaptive Mesh so it can capture good threshold 
        tot = 0
        i = 0
        while i < 3:
            tpt = self.tpoints[key][randint(0, len(self.tpoints[key]))]
            if self.dim == 2:
                tot += getDistance2D(tpt,tree.getNearR(tpt,[]))
            else:
                tot += getDistance3D(tpt,tree.getNearR(tpt,[]))
            i += 1
        self.threshDistance.append(tot / 3)
        
        ##START OF SOLVE
        index = 0
        guessr = 0
        prnd = []
        for point in self.tpoints[key]:
            #finding inital temp radius
            norm = self.tnorms[key][index]
            tempr = []
            if index == 0:
                prnd = self.tpoints[key][randint(1, len(self.tpoints[key]))]
                tempr.append(np.round(getRadius2D(point,prnd,norm),6))
            else:
                tempr.append(guessr)
            i = 0
            centerp = []
            centerp.append([float(point[0]-norm[0]*tempr[0]),float(point[1]-norm[1]*tempr[0])])
            testp = []
            testp.append(prnd)
            case = False
            #Main loop for each points solve
            while not case:
                testp.append(tree.getNearR(centerp[len(centerp) - 1],point))
                if self.dim == 2:
                    tempr.append(np.round(getRadius2D(point,testp[index + 1],norm),6))                
                    centerp.append([float(point[0]-norm[0]*tempr[i+1]),float(point[1]-norm[1]*tempr[i+1])])
                else:
                    tempr.append(np.round(getRadius3D(point,testp[index + 1], norm),6))
                    centerp.append([float(point[0]-norm[0]*tempr[i+1]),float(point[1]-norm[1]*tempr[i+1]),float(point[2]-norm[2]*tempr[i+1])])
                leng = len(tempr) - 1
                if i > 1 and np.abs(tempr[leng] - tempr[leng - 1]) < 0.00001:
                    
                
                i += 1
            
            
            index += 1
                
                
                
    def tag(self):
        i = 0
        #Organizing Points means going through and seperating everything by tag
        while i < len(self.IntPoints):
            if i > 0:
                found = False
                j = 0
                for tag in self.tagKey:
                    size = len(self.tagKey)
                    if self.MasterTag[i] == tag:
                        found = True
                        self.tpoints[j].append(self.IntPoints[i])
                        self.tnorms[j].append(self.NormPoints[i])
                    j += 1
                if not(found):
                    self.tagKey.append(self.MasterTag[i])
                    self.tpoints.append([])
                    self.tnorms.append([])
                    self.tpoints[size].append(self.IntPoints[i])
                    self.tnorms[size].append(self.NormPoints[i])  
            else:  
                self.tagKey.append(self.MasterTag[i])
                self.tpoints.append([])
                self.tnorms.append([])
                self.tpoints[0].append(self.IntPoints[i])
                self.tnorms[0].append(self.NormPoints[i])
            i += 1
        
###MISC FUCNTIONS FOR SKELENET
    
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

    def plot(self,mode : [1] = [],*,norm = True):
        index = 0
        tt = 0
        while index < len(mode)
            print("Plotting {}".format(mode[index]))
            #Mode 0 -> output to degbug of normals of each point
            if mode == 0:
                st = time.time()
                tx = []
                ty = []
                i = 0
                while i < len(self.IntPoints):
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                i = 0
                while i < len(self.IntPoints):
                    print(i,'/',len(self.IntPoints) - 1)
                    plt.xlim(-0.3,0.2)
                    plt.ylim(-0.3,0.2)
                    plt.scatter(tx,ty)
                    plt.scatter(self.IntPoints[i][0],self.IntPoints[i][1]) 
                    if norm == False:
                        plt.plot([self.IntPoints[i][0] + self.NormPoints[i][0] * 1000,self.IntPoints[i][0] + self.NormPoints[i][0] * - 1000],[self.IntPoints[i][1] + self.NormPoints[i][1] * 1000,self.IntPoints[i][1] + self.NormPoints[i][1] * -1000])
                    else:
                        plt.plot([self.IntPoints[i][0] + self.NormPoints[i][0] * 1000,self.IntPoints[i][0]],[self.IntPoints[i][1] + self.NormPoints[i][1] * 1000,self.IntPoints[i][1]])
                    save = os.getcwd() + "\Debug\Debug{:04d}.png".format(i)
                    plt.savefig(save)
                    plt.clf()
                    i += 1
                et = time.time()
                tt += (et - st)
            #Mode 1 is for outputting final points
            elif mode == 1:
            
            
            
        
        
        
        
        print('Animation took {} minuites and {} seconds'.format((tt) // 60,(tt) % 60))
                