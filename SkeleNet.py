import os
from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import csv
import scipy
import pandas as pd
import time

from DataStructures import kdTree, SkelePoint
from Skeletize import checkRepeat,getRadius,getDistance,normalize

class SkeleNet:
    #In simpleTerms Skelenet is an easy to use skeletonization processer, 
    #It can intake a location of a data file, or even the straight points
    #Then itll process then and output different figures for different things
    #Can also Produce different shapes and stuff
    

    rnd = 0
###INITALIZERS
    def __init__(self, points,*,norms = []):
        #Solve variables, certain parts of these can be thrown away later on depending id heap is heavy
        self.IntPoints = []
        self.NormPoints = []
        self.MasterTag = []
        self.tagKey = []
        self.tpoints = []
        self.tnorms = []
        self.threshDistance = []
        #Final Variables (for now depending on how we later edit this information)
        self.SkelePoints = []
        self.SkeleRad = []
        self.tagged = False
        #Determining type of points given
        if isinstance(points,str):    
            with open(points,'r') as csvfile:
                data = csv.reader(csvfile, delimiter = ' ')
                for row in data:
                    
                    size = len(row)
                    if str(row[0]) == 'x':#if title
                        a = 1    
                    elif size == 4:#2D w/ no tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1])])
                            self.NormPoints.append([float(row[2]),float(row[3])])
                    elif size == 5:#2D w/ tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1])])
                            self.NormPoints.append([float(row[2]),float(row[3])]) 
                            self.MasterTag.append(int(row[4]) - 1)
                            self.tagged = True
                    elif size == 6:#3D w/ no tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                            self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
                    elif size == 7:#3D w/ tag
                        if randint(0,10) >= self.rnd:
                            self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                            self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
                            self.MasterTag.append(int(row[6]) - 1)
                            self.tagged =  True
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

        temp  = normalize(self.NormPoints)
        self.NormPoints = []
        self.NormPoints = temp
        
        if self.tagged:
            self.__tag()
        else:
            self.tpoints.append(self.IntPoints)
            self.tnorms.append(self.NormPoints)

###MAIN FUNCTIONS
    def solve(self,animate : bool = False):
        #Solves for all taggs individually
        #will be paralled in the future
        if animate:
            self.animate = True
            self.acp = []
            self.atp = []
            self.arad = []
        else:
            self.animate = False
        
        if self.tagged:
            i = 0
            while i < len(self.tpoints):
                self.__skeletize(i)
                i += 1
        else:
            self.__skeletize(0)
        
    def order(self):
        #This function will go through all of the points 
        t = 0
        self.OPoints = []#Ordered Points(Contain radius for later)
        while t < len(self.SkelePoints):
            i = 0
            self.OPoints.append([])
            while i < len(self.SkelePoints[t]):
                self.OPoints[t].append(SkelePoint(self.SkelePoints[t][i],self.SkeleRad[t][i]))
                i += 1
            t += 1
        
        #Orders Them Now.
        t = 0
        while t < len(self.OPoints): 
            i = 0
            t += 1
        
    def __skeletize(self,key : int):
        #Skeletize takes in 
        #FROM INPUT
        #key is the specific index of tpoints and tnorms, allows for
        #parallel capabilities in splitting apart skeleton tasks 
        #Also allows the class to have an array being filled by different tags
        #FROM CLASS
        #points, the given plain list of points [x,y] for 2D case
        #norms, a list of not yet normalized normal points [n_x,n_y] here for 2D case
        #then returns 2 things
        # finPoints = [[x1,y1],...] of skeleton points
        # finR = [r1,...] of the radius of each skeleton point
        ts = time.time() 
        self.SkelePoints.append([])
        self.SkeleRad.append([])
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
            tpt = self.tpoints[key][randint(0, len(self.tpoints[key]) - 1)]
            tot += getDistance(tpt,tree.getNearR(tpt,[]))
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
            i = 0
            centerp = []
            testp = []
            
            case = False
            
            #Setup animate
            if self.animate:
                if index ==  0:
                    self.acp.append([])
                    self.atp.append([])
                    self.arad.append([])
            #Main loop for each points solve
            while not case:
                if i == 0:
                    #Inital
                    if index == 0:
                        prnd = self.tpoints[key][randint(1, len(self.tpoints[key]) - 1)]
                        tempr.append(np.round(getRadius(point,prnd,norm),6))
                    else:
                        tempr.append(guessr)
                        
                    if self.dim == 2:
                        centerp.append([float(point[0]-norm[0]*tempr[0]),float(point[1]-norm[1]*tempr[0])])
                    else:
                        centerp.append([float(point[0]-norm[0]*tempr[0]),float(point[1]-norm[1]*tempr[0]),float(point[2]-norm[2]*tempr[0])])
                    testp.append(point)
                else:   
                    #Refinement of skeleton point
                    testp.append(tree.getNearR(centerp[len(centerp) - 1],point))
                    tempr.append(np.round(getRadius(point,testp[i],norm),6))
                    if self.dim == 2:
                        centerp.append([float(point[0]-norm[0]*tempr[i]),float(point[1]-norm[1]*tempr[i])])
                    else:
                        centerp.append([float(point[0]-norm[0]*tempr[i]),float(point[1]-norm[1]*tempr[i]),float(point[2]-norm[2]*tempr[i])])
                leng = len(tempr) - 1
                
                #Capture animation data
                if self.animate:
                    if i == 0:
                        self.acp[key].append([])
                        self.atp[key].append([])
                        self.arad[key].append([])
                        self.acp[key][index].append(centerp[0])
                        self.atp[key][index].append(testp[0])
                        self.arad[key][index].append(tempr[0])
                    self.acp[key][index].append(centerp[leng])
                    self.atp[key][index].append(testp[leng])
                    self.arad[key][index].append(tempr[leng])
                    
                #Checking for completeion
                
                #Convergence check
                if i > 1 and np.abs(tempr[leng] - tempr[leng - 1]) < self.threshDistance[key]:
                    if tempr[leng] < (self.threshDistance[key] / 2) or getDistance(point, testp[leng]) < tempr[leng]:
                        self.SkelePoints[key].append(centerp[leng - 1])
                        self.SkeleRad[key].append(tempr[leng - 1])
                        #Show backstep in animation
                        if self.animate:
                            self.acp[key][index].append(centerp[leng - 1])
                            self.atp[key][index].append(testp[leng - 1])
                            self.arad[key][index].append(tempr[leng - 1])
                    else:
                        self.SkelePoints[key].append(centerp[leng])
                        self.SkeleRad[key].append(tempr[leng])
                    
                    case = True 
                
                #Overshooting  
                elif i > 1 and tempr[leng] < (self.threshDistance[key] / 2):
                    self.SkelePoints[key].append(centerp[leng - 1])
                    self.SkeleRad[key].append(tempr[leng - 1])
                    #Show backstep in animation
                    if self.animate:
                        self.acp[key][index].append(centerp[leng - 1])
                        self.atp[key][index].append(testp[leng - 1])
                        self.arad[key][index].append(tempr[leng - 1])
                    case = True
                elif i > 1 and getDistance(point, testp[leng]) < tempr[leng]:
                    self.SkelePoints[key].append(centerp[leng - 1])
                    self.SkeleRad[key].append(tempr[leng - 1])
                    #Show backstep in animation
                    if self.animate:
                        self.acp[key][index].append(centerp[leng - 1])
                        self.atp[key][index].append(testp[leng - 1])
                        self.arad[key][index].append(tempr[leng - 1])
                    case = True
                
                
                #Repeat check
                elif i > 3:
                    #Really only comes up with perfect shapes,
                    #but always a possibility to happen
                   repeat, order = checkRepeat(tempr)
                   if repeat:
                       n = 0
                       p = 0
                       sml = 0.0
                       while p < order:
                           if p == 0:
                               sml = tempr[len(tempr) - (order)]
                           else:
                               tmp = tempr[len(tempr)-(order - p)]
                               if tmp < sml:
                                   sml = tempr[len(tempr)-(order-p)]
                                   n = len(tempr) - (order - p)
                           p = p + 1
                       print('Repeat')
                       self.SkeleRad[key].append(sml)
                       self.SkelePoints[key].append(centerp[n])
                       case = True 
                i += 1
            if index != len(self.tpoints[key]) - 1:
                guessr = self.threshDistance[key] * len(self.tpoints[key])
            index += 1
        te = time.time()
        tt = te - ts
        print('Skeleton  #{} took {} minuites and {} seconds'.format(key,(tt) // 60,(tt) % 60))
                
                
                
    
        
###MISC FUCNTIONS FOR SKELENET
    def __tag(self):
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
            
    def __getNorms(self):
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

    def plot(self,mode : list = [],*,norm = True,tag = 'None',start : int = 0,stop : int = 9999):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        theta = np.linspace(0,2*np.pi,100)
        
        index = 0
        tt = 0
        while index < len(mode):
            print("Plotting {}".format(mode[index]))
            
            #Mode 0 -> output to degbug of normals of each point
            if mode[index] == 0:
                plt.clf()
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
                
            #Mode 1 is for outputting final points for every tag
            elif mode[index] == 1:
                plt.clf()
                st = time.time()
                if self.tagged:
                    i = 0
                    tx = []
                    ty = []
                    while i < len(self.IntPoints):    
                        tx.append(self.IntPoints[i][0])
                        ty.append(self.IntPoints[i][1])
                        i += 1
                    plt.scatter(tx,ty,5,color='blue')
                    i = 0
                    #theta = np.linspace(0,2*np.pi)
                    while i < len(self.SkelePoints):
                        j = 0
                        tx = []
                        ty = []
                        while j < len(self.SkelePoints[i]):
                            tx.append(self.SkelePoints[i][j][0])
                            ty.append(self.SkelePoints[i][j][1])
                            #plt.plot(tx[j] + np.cos(theta) * self.SkeleRad[i][j],ty[j] + np.sin(theta) * self.SkeleRad[i][j],color = 'blue')
                            j += 1
                        plt.scatter(tx,ty,5)
                        i += 1
                else:
                    i = 0
                    tx = []
                    ty = []
                    while i < len(self.IntPoints):    
                        tx.append(self.IntPoints[i][0])
                        ty.append(self.IntPoints[i][1])
                        i += 1
                    plt.scatter(tx,ty,5,color='blue')
                    i = 0
                    tx = []
                    ty = []
                    #theta = np.linspace(0,2*np.pi)
                    while i < len(self.SkelePoints[0]):
                        tx.append(self.SkelePoints[0][i][0])
                        ty.append(self.SkelePoints[0][i][1])
                        #plt.plot(tx[i] + np.cos(theta) * self.SkeleRad[0][i],ty[i] + np.sin(theta) * self.SkeleRad[0][i],color = 'blue')
                        i += 1
                    plt.scatter(tx,ty,5)
                plt.savefig('Output.png')
                et = time.time()
                tt += (et - st)
            #Mode2 is for Animating the process of solving
            elif mode[index] == 2:
                st = time.time()
                svnum = 0
                plt.clf()
                path = os.getcwd()
                tag = 0
                i = 0
                case = True
                path = path + "/AnimationData/"
                while case:
                    tpath = path + f'{i:04d}' + '/'
                    if not(os.path.isdir(tpath)):
                        case = False
                        path = tpath
                        os.mkdir(tpath)
                    i += 1
                i = 0
                tx = []
                ty = []
                while i < len(self.IntPoints):    
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                sx = []
                sy = []
                while tag < len(self.acp):
                    i = start
                    while i < len(self.acp[tag]):
                        j = 0
                        while j < len(self.acp[tag][i]):
                            #print(tag, '/', len(self.acp),' ', i '/' , len(self.acp[tag], ' ', j ))
                            plt.clf()
                            plt.xlim(0,0.5)
                            plt.ylim(0,0.5)
                            plt.scatter(tx,ty,5,color='green')
                            if len(sx) > 0:
                                plt.scatter(sx,sy,5,color='orange')
                            plt.plot([self.acp[tag][i][j][0],self.tpoints[tag][i][0]],[self.acp[tag][i][j][1],self.tpoints[tag][i][1]])
                            plt.plot([self.atp[tag][i][j][0],self.tpoints[tag][i][0]],[self.atp[tag][i][j][1],self.tpoints[tag][i][1]])
                            plt.plot(self.acp[tag][i][j][0] + np.cos(theta) * self.arad[tag][i][j],self.acp[tag][i][j][1] + np.sin(theta) * self.arad[tag][i][j])
                            plt.scatter(self.acp[tag][i][j][0],self.acp[tag][i][j][1],5,color='purple')
                            plt.scatter(self.atp[tag][i][j][0],self.atp[tag][i][j][1],5,color='red')
                            plt.scatter(self.tpoints[tag][i][0],self.tpoints[tag][i][1],5,color='blue')
                            plt.title('{},radius : {}, distance : {}'.format(i,self.arad[tag][i][j],getDistance(self.tpoints[tag][i],self.atp[tag][i][j])))
                            
                            plt.savefig(path + '{:04d}.png'.format(svnum))
                            svnum += 1
                            j += 1
                        sx.append(self.acp[tag][i][j - 1][0])
                        sy.append(self.acp[tag][i][j - 1][1])
                        i += 1
                        if i == stop:
                            break
                    tag += 1
                    
                et = time.time()
                tt += (et - st)
            index += 1        

            
            
        
        
        
        
        print('Animation took {} minuites and {} seconds'.format((tt) // 60,(tt) % 60))
                
