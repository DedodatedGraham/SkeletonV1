# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:46:08 2022

@author: graha
"""
# from random  import randint
# from sys import float_repr_style
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits import mplot3d
import numpy as np
import time
# import csv
# import scipy
# import pandas as pd
# import numpy as np

from Skeletize import getDistance, getAngle

# @profile
def quicksort(points : list , dimension : int,*,cpuavail : int = 1):
    #  **Points are Skeleton Points**
    return quicksortrunner(points,dimension,0,len(points)-1,cpuavail=cpuavail)

# @profile
def quicksortrunner(points : list , dimension : int , first : int , last : int ,depth : int = 0 , * , cpuavail : int = 1):
    # print('sorting',depth)
    if first<last:
        splitpoint = partition(points,dimension,first,last,cpuavail=cpuavail)
        quicksortrunner(points, dimension, first, splitpoint - 1,depth=depth+1,cpuavail=cpuavail)
        quicksortrunner(points, dimension, splitpoint + 1,last,depth=depth+1,cpuavail=cpuavail)
        return points
# @profile           
def partition(points : list , dimension : int , first : int , last : int , * , cpuavail : int = 1 ):
    pivot = points[first].getAxis(dimension)
    left = first + 1
    right = last
    done = False
    while not done:
        while left <= right and points[left].getAxis(dimension) <= pivot:
            left += 1
        while right >= left and points[right].getAxis(dimension) >= pivot:
            right -= 1
        if right < left:
            done = True
        else:
            temp = points[left]
            points[left] = points[right]
            points[right] = temp
    temp = points[first]
    points[first] = points[right]
    points[right] = temp
    return right





class kdTree:
    # @profile
    def __init__(self , points : list , depth : int = 0 , * , rads : list = [],dimensions : int = 0,cpuavail : int = 1):
        #print('making {}'.format(depth))
        #first converts to skeleton points
        if not(dimensions == 0):
            self.dimensions = dimensions
        if depth == 0 and not(isinstance(points[0],SkelePoint)):
            st = time.time()
            print('Initiating k-d,Tree')
            self.dimensions = len(points[0])
            i = 0
            tp = []
            while i < len(points):
                if len(rads) > 0:
                    tp.append(SkelePoint(points[i], rad=rads[i]))
                else:
                    tp.append(SkelePoint(points[i]))
                i += 1
            points = tp
            # ett = time.time()
            # ttt = ett - st
            # print('point conversion took {} minuites and {} seconds to make {} SkelePoints'.format(ttt // 60,ttt % 60, len(points)))
        self.depth = depth
        #Next is the actual process of creating the struct
        if len(points) > 100:
            self.split = True
            self.axis = depth % self.dimensions
            points = quicksort(points,self.axis,cpuavail=cpuavail)
            # ttt = time.time() -  stt
            # print('quicksearch took {} minuites and {} seconds to sort {} SkelePoints'.format(ttt // 60,ttt % 60, len(points)))
            mid = len(points) // 2
            self.node = points[mid]
            i = 0
            pointsl = []
            while i < mid:
                pointsl.append(points[i])
                i += 1
            self.leafL = kdTree(pointsl,depth + 1,dimensions=self.dimensions)
            i = mid + 1
            pointsr = []
            while i < len(points):
                pointsr.append(points[i])
                i += 1
            self.leafR = kdTree(pointsr,depth + 1,dimensions=self.dimensions)
        else:
            self.split = False
            i = 0
            self.points = []
            while i < len(points):
                self.points.append(points[i])
                i += 1
        if depth == 0:
            et = time.time()
            tt = et - st
            print('k-d tree took {} minuites and {} seconds to make'.format(tt // 60,tt % 60))
    # @profile
    def getNearR(self,inputdat : list):
        if len(inputdat) > 2:
            getRads = inputdat[2]
            if len(inputdat) > 3:
                cpuavail = inputdat[3]
            tdat = []
            tdat.append(inputdat[0])
            tdat.append(inputdat[1])
            inputdat = tdat
                
        # print('searching...')
        dmin = 100000
        if self.split:
            pmin = self.node
            #If needs to go further
            if inputdat[0][self.axis] <= self.node.getAxis(self.axis):
                tpmin,dmin =  self.leafR.getNearR(inputdat)
                if tpmin != 0:
                    pmin = tpmin
                #Next checks node point and sees if its closer, overwrites if so
                if not(self.node.getPoint() == inputdat[0]) and not(self.node.getPoint() == inputdat[1]):
                    ndis = getDistance(inputdat[0], self.node.getPoint())
                    if ndis < dmin:
                        pmin = self.node
                        dmin =  ndis
                tdis = self.node.getAxis(self.axis) - inputdat[0][self.axis]
                if tdis <= dmin:
                    pmin1,dmin1 =  self.leafL.getNearR(inputdat)
                    if not(pmin1 == 0) and not(pmin1.getPoint() == inputdat[0]) and not(pmin1.getPoint() == inputdat[1]):
                        if dmin1 < dmin:
                            dmin = dmin1
                            pmin = pmin1
            else:
                tpmin,dmin =  self.leafL.getNearR(inputdat) 
                if tpmin != 0:
                    pmin = tpmin
                #Next checks node point and sees if its closer, overwrites if so
                if not(self.node.getPoint() == inputdat[0]) and not(self.node.getPoint() == inputdat[1]):
                    ndis = getDistance(inputdat[0], self.node.getPoint())
                    if ndis < dmin:
                        pmin = self.node
                        dmin =  ndis
                tdis = inputdat[0][self.axis] - self.node.getAxis(self.axis)
                if tdis <= dmin:
                    pmin1,dmin1 =  self.leafR.getNearR(inputdat)
                    if not(pmin1 == 0) and not(pmin1.getPoint() == inputdat[0]) and not(pmin1.getPoint() == inputdat[1]):
                        if dmin1 < dmin and not(dmin1) == 0:
                            dmin = dmin1
                            pmin = pmin1
                            
        
        else:
            #If at the lowest
            i = 0
            while i < len(self.points):
                if i == 0:
                    if not(self.points[i].getPoint() == inputdat[1]) and not(self.points[i].getPoint() == inputdat[0]):
                        dmin = getDistance(inputdat[0],self.points[i].getPoint())
                        pmin = self.points[i]
                else:
                   if not(self.points[i].getPoint() == inputdat[1]) and not(self.points[i].getPoint() == inputdat[0]):
                       tmin = getDistance(inputdat[0],self.points[i].getPoint())
                       if tmin < dmin:
                           dmin = tmin
                           pmin = self.points[i]
                i += 1
            if not('pmin' in locals()):
                pmin = 0
        if self.depth == 0:
            if getRads:
                return pmin.getPoint(),pmin.getRad()
            else:
                return pmin.getPoint()
        else:
            return pmin,dmin
    # @profile    
    def getVectorR(self,inputdat : list,*,depth : int = 0,scan = np.pi / 4,cpuavail : int = 1):
            #Get vector will get the closest n number of points to the search point
            #it consideres a 'scan' degree area along the given vector, will go deepest first as thats where the closest points should be
            #however it will store node value and compare on the way out if a given node is a better point than something returned
            
            if depth == 0:
                if len(inputdat) > 3:
                    getRads = inputdat[3]
                    if len(inputdat) > 4:
                        scan = inputdat[4]
                        if len(inputdat) > 5:
                            cpuavail = inputdat[5]
                tin = []
                tin.append(inputdat[0])
                tin.append([])
                i = 0
                while i < len(inputdat[1]):
                    tin[1].append(inputdat[1][i])
                    i += 1
                tin.append(inputdat[2])
                inputdat = tin
            retPoints = []
            retDist = []
            axis = depth % self.dimensions
            steps = ''
            if not(self.split):
                i = 0
                while i < len(self.points):
                    #each point it first checks the angle between the vectors.
                    j = 0
                    cvec = []
                    tpoint = []
                    while j < len(inputdat[1]):
                        cvec.append(self.points[i].getAxis(j) - inputdat[0][j])
                        tpoint.append(inputdat[0][j] + inputdat[1][j])
                        j += 1
                    tdis = getDistance(inputdat[0],self.points[i].getPoint())
                    theta = getAngle(inputdat[1],cvec,getDistance(inputdat[0],tpoint),tdis)
                    if theta  < (scan / 2):
                        #Within vector distance
                        if len(retPoints) < inputdat[2]:
                            retPoints.append(self.points[i])
                            retDist.append(tdis)
                        else:
                            j = 0
                            tagj = 0
                            mtdis = -1
                            while j < len(retPoints):
                                if tdis < retDist[j] and retDist[j] > mtdis:
                                    tagj = j
                                    mtdis = retDist[j]
                                j += 1
                            if not(mtdis == -1):
                                retPoints[tagj] = self.points[i]
                                retDist[tagj] = tdis 
                    i += 1
            else:
                node = self.node
                #First want to see if the vector only reaches a specific leaf of the tree
                i = 0
                # vecax = []
                tp1 = []
                # tp2 = []
                nvec = []
                while i < len(inputdat[0]):
                    # if i == axis:
                    #     vecax.append(1)
                    # else:
                    #     vecax.append(0)
                    tp1.append(inputdat[0][i] + inputdat[1][i])
                    # tp2.append(inputdat[0][i] + vecax[i])
                    nvec.append(node.getAxis(i) - inputdat[0][i])
                    i += 1
                # theta = getAngle(inputdat[1],vecax,getDistance(inputdat[0],tp1),getDistance(inputdat[0],tp2))
                # mat = []
                # if self.dimensions == 2:
                #     if axis == 0:
                #         mat.append(inputdat[1][0] * np.cos(scan / 2) - inputdat[1][1] * np.sin(scan / 2))
                #         mat.append(inputdat[1][0] * np.cos(-scan / 2) - inputdat[1][1] * np.sin(-scan / 2))
                #     else:
                #         mat.append(inputdat[1][0] * np.sin(scan / 2) + inputdat[1][1] * np.cos(scan / 2))
                #         mat.append(inputdat[1][0] * np.sin(-scan / 2) +inputdat[1][1] * np.cos(-scan / 2))
                # else:
                #     if axis == 0:
                #         #X axis rotation
                #         mat.append([])
                #         mat[0].append(inputdat[1][0] * np.cos(scan / 2) - inputdat[1][1] * np.sin(scan / 2))
                #         mat[0].append(inputdat[1][0] * np.cos(-scan / 2) - inputdat[1][1]  * np.sin(-scan / 2))
                #         mat.append([])
                #         mat[1].append(inputdat[1][0] * np.sin(scan / 2) + inputdat[1][1] * np.cos(scan / 2))
                #         mat[1].append(inputdat[1][0] * np.sin(-scan / 2) + inputdat[1][1] * np.cos(-scan / 2))
                #         mat.append([])
                #         mat[2].append(inputdat[1][2])
                #         mat[2].append(inputdat[1][2])
                #     elif axis == 1:
                #         #Y axis rotation
                #         mat.append([])
                #         mat[0].append(inputdat[1][0] * np.cos(scan / 2) + inputdat[1][2] * np.sin(scan / 2))
                #         mat[0].append(inputdat[1][0] * np.cos(-scan / 2) + inputdat[1][2]  * np.sin(-scan / 2))
                #         mat.append([])
                #         mat[1].append(inputdat[1][1])
                #         mat[1].append(inputdat[1][1])
                #         mat.append([])
                #         mat[2].append(-inputdat[1][0] * np.sin(scan / 2) + inputdat[1][2] * np.cos(scan / 2))
                #         mat[2].append(-inputdat[1][0] * np.sin(-scan / 2) + inputdat[1][2] * np.cos(-scan / 2))
                #     else:
                #         #Z axis rotation
                #         mat.append([])
                #         mat[0].append(inputdat[1][0])
                #         mat[0].append(inputdat[1][0])
                #         mat.append([])
                #         mat[1].append(inputdat[1][1] * np.cos(scan / 2) - inputdat[1][2] * np.sin(scan / 2))
                #         mat[1].append(inputdat[1][1] * np.cos(-scan / 2) - inputdat[1][2] * np.sin(-scan / 2))
                #         mat.append([])
                #         mat[2].append(inputdat[1][1] * np.sin(scan / 2) + inputdat[1][2] * np.cos(scan / 2))
                #         mat[2].append(inputdat[1][1] * np.sin(-scan / 2) + inputdat[1][2] * np.cos(-scan / 2))
                # if depth == 0:
                #     print()
                #     print(mat,theta)
                #     print(inputdat[0],inputdat[1])
                
                
                if node.getAxis(axis) < inputdat[0][axis] and inputdat[1][axis] > 0.1:
                    retPoints,retDist,steps = self.leafR.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
                    if depth < 4 and len(retPoints) == 0:
                        if depth == 0:
                            print()
                            print('went left and shouldnt of ?',depth)
                            print('point',inputdat[0],'norm',inputdat[1])
                            print('node',node.getPoint())
                            print('axis',axis)
                            print('steps:',steps)
                            print()
                        else:
                            steps += 'went left and shouldnt of?'
                            steps += str(node.getPoint())
                            steps += str(depth)
                            
                elif node.getAxis(axis) > inputdat[0][axis] and inputdat[1][axis] < 0.1:
                    retPoints,retDist,steps = self.leafL.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
                    if depth < 4 and len(retPoints) == 0:
                        if depth == 0:
                            print()
                            print('went right and shouldnt of ?',depth)
                            print('point',inputdat[0],'norm',inputdat[1])
                            print('node',node.getPoint())
                            print('axis',axis)
                            print(steps)
                            print()
                        else:
                            steps += 'went right and shouldnt of ?'
                            steps += str(node.getPoint())
                            steps += str(depth)
                else:
                    # print('bruh',mat[axis])
                    tretPoints = []
                    tretDist = []
                    retPointsl = []
                    retPointsr = []
                    retDistl = []
                    retDistr = []
                    retPointsl,retDistl,stepsl = self.leafL.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
                    retPointsr,retDistr,stepsr = self.leafR.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
                    steps += stepsl
                    steps += stepsr
                    if depth < 4 and len(retPointsl) == 0 and len(retDistr) == 0:
                        if depth == 0:
                            print()
                            print('went Both and got nothin?',depth)
                            print('point',inputdat[0],'norm',inputdat[1])
                            print('node',node.getPoint())
                            print('axis',axis)
                            print(steps)
                            print()
                        else:
                            steps += 'went Both and got nothin?'
                            steps += str(node.getPoint())
                            steps += str(depth)
                            
                    i = 0
                    while i < len(retPointsl):
                        tretPoints.append(retPointsl[i])
                        tretDist.append(retDistl[i])
                        i += 1
                    i = 0
                    while i < len(retPointsr):
                        tretPoints.append(retPointsr[i])
                        tretDist.append(retDistr[i])
                        i += 1
                    ntheta = getAngle(inputdat[1],nvec,getDistance(inputdat[0],tp1),getDistance(inputdat[0],node.getPoint()))
                    #gets the node point if it falls inside the criteria
                    if ntheta < (scan / 2): 
                        tretPoints.append(node)
                        tretDist.append(getDistance(inputdat[0],node.getPoint()))
                    #Finally aquires the best n# of points from big list
                    i = 0
                    while i < len(tretPoints):
                        if i < inputdat[2]:
                            retPoints.append(tretPoints[i])
                            retDist.append(tretDist[i])
                        else:
                            tdis = tretDist[i]
                            j = 0
                            tagj = 0
                            mtdis = -1
                            while j < len(retPoints):
                                
                                if tdis < retDist[j] and retDist[j] > mtdis:
                                    tagj = j
                                    mtdis = retDist[j]
                                j += 1
                            if not(mtdis == -1):
                                retPoints[tagj] = tretPoints[i]
                                retDist[tagj] = tdis 
                        i += 1
                    
            if depth == 0:
                j = 0
                trp = []
                trad = []
                while j < len(retPoints):
                    trp.append(retPoints[j].getPoint())
                    if getRads:
                        trad.append(retPoints[j].getRad())
                    j += 1
                if getRads:
                    return trp, trad
                else:
                    return trp
            else:
                return retPoints,retDist,steps
                    
                
     def getInR(self,point : list, dim : float, mode : int,depth : int = 0,*,getRads : bool = False):
         #Returns all the points which lie inside a give area arround a certain point
         #Mode 0 => Square area, point in center, side = 2 * dim
         #Mode 1 => Circle area, point in center, rad  = dim
         retPoints = []
         axis = depth % self.dimensions
         node = self.node        
         if not(self.split):
             i = 1
             while i < len(tree):
                 if mode == 0:
                     if tree[i][0] >= point[0] - dim and tree[i][0] <= point[0] + dim:
                         if tree[i][1] >= point[1] - dim and tree[i][1] <= point[1] + dim:
                             if self.dmensions == 2 or (self.dimensions == 3 and tree[i][2] >= point[2] - dim and tree[i][2] <= point[2] + dim):
                                 retPoints.append(tree[i])
                 else:
                     if getDistance(point,tree[i]) <= dim:
                         retPoints.append(tree[i])
                 i += 1
         else:
             #Uses square to obtain all possible sections that might be needed
             #mode only comes into play when searching at bottom layers, however adding the node points 
             #will still depend on mode  
             pts = []
             rads = []
             if point[axis] - dim > node[axis]:
                 if getRads:
                     pts,rads = self.getInR(point,dim,mode,tree[2],depth + 1, getRads=True,rtree = rtree[2])
                 else:
                     pts = self.getInR(point,dim,mode,tree[2],depth + 1)
             elif point[axis] + dim < node[axis]:
                 if getRads:
                     pts,rads = self.getInR(point,dim,mode,tree[1],depth + 1, getRads = True, rtree = rtree[1])
                 else:
                     pts = self.getInR(point,dim,mode,tree[1],depth + 1)
             else:
                 if getRads:
                     pts1,rad1 = self.getInR(point,dim,mode,tree[1],depth + 1, getRads = True, rtree = rtree[1])
                     pts2,rad2 = self.getInR(point,dim,mode,tree[2],depth + 1, getRads = True, rtree = rtree[2])
                 else:
                     pts1 = self.getInR(point,dim,mode,tree[1],depth + 1)
                     pts2 = self.getInR(point,dim,mode,tree[2],depth + 1)
                 i = 0
                 while i < len(pts1):
                     pts.append(pts1[i])
                     if getRads:
                         rads.append(rad1[i])
                     i += 1
                 i = 0
                 while i < len(pts2):
                     pts.append(pts2[i])
                     if getRads:
                         rads.append(rad2[i])
                     i += 1
                 #Note only needs to check node here
                 if mode == 0:
                     if node[0] >= point[0] - dim and node[0] <= point[0] + dim:
                         if node[1] >= point[1] - dim and node[1] <= point[1] + dim:
                             if self.dmensions == 2 or (self.dimensions == 3 and node[2] >= point[2] - dim and node[2] <= point[2] + dim):
                                 pts.append(node)
                                 if getRads:
                                     rads.append(rnode)
                 else:
                     if getDistance(point,node) <= dim:
                         pts.append(node)
                         if getRads:
                             rads.append(rnode)
               
             if len(pts) > 0:
                 i = 0
                 while i < len(pts):
                     retPoints.append(pts[i])
                     if getRads:
                         retR.append(rads[i])
                     i += 1
         if getRads:
             return retPoints,retR
         else:
             return retPoints

class SplitTree:
    #Split Tree is a versatile Quad/Oct tree designed for efficient stack storage for search
    def __init__(self,inpts,node:list,width: float,*,inrad : list = [],dim : int = 0):
        #Bounds are stored in a node(center [x,y,z]), heigh and width
        self.count = len(inpts)
        self.state = False
        if not(dim == 2 or dim == 3):
            if isinstance(inpts[0],SkelePoint):
                self.dim = inpts[0].dimensions
            else:
                if len(inpts[0]) == 2:
                    self.dim = 2
                else:
                    self.dim = 3
        else:
            self.dim = dim
        if self.dim == 2:
            self.maxpts = 4
        else:
            self.maxpts = 8
        #Defining skele Points
        self.skelepts = []
        if not(len(inpts) == 0):
            if isinstance(inpts[0],list):
                i = 0
                while i < len(inpts):
                    if len(inrad) > 0:
                        self.skelepts.append(SkelePoint(inpts[i],rad = inrad[i]))    
                    else:
                        self.skelepts.append(SkelePoint(inpts[i]))
                    i += 1
            else:
                i = 0
                while i < len(inpts):
                    self.skelepts.append(inpts[i])
                    i += 1
        #Defining other important elements
        self.node = node
        self.width = width
        #If there are too many points, will subdivide
        if len(self.skelepts) > self.maxpts:
            self.state = True
            self.subdivide()
            
    def subdivide(self):
        #Creates the nodes for the new Quad/oct trees, and sorts points to their appropiate node
        self.leafs = []
        nodes = []
        points = []
        if self.dim == 2:
            nodes.append([self.node[0] + 0.5 * self.width,self.node[1] + 0.5 * self.width])
            nodes.append([self.node[0] + 0.5 * self.width,self.node[1] - 0.5 * self.width])
            nodes.append([self.node[0] - 0.5 * self.width,self.node[1] + 0.5 * self.width])
            nodes.append([self.node[0] - 0.5 * self.width,self.node[1] - 0.5 * self.width])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            i = 0
            while i < len(self.skelepts):
                if self.skelepts[i].x > self.node[0] and self.skelepts[i].y > self.node[1]:
                    points[0].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y < self.node[1]:
                    points[1].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y > self.node[1]:
                    points[2].append(self.skelepts[i])
                else:
                    points[3].append(self.skelepts[i])
                i += 1
        else:
            nodes.append([self.node[0] + 0.5 * self.width,self.node[1] + 0.5 * self.width,self.node[2] + 0.5 * self.width])
            nodes.append([self.node[0] + 0.5 * self.width,self.node[1] + 0.5 * self.width,self.node[2] - 0.5 * self.width])
            nodes.append([self.node[0] + 0.5 * self.width,self.node[1] - 0.5 * self.width,self.node[2] + 0.5 * self.width])
            nodes.append([self.node[0] + 0.5 * self.width,self.node[1] - 0.5 * self.width,self.node[2] - 0.5 * self.width])
            nodes.append([self.node[0] - 0.5 * self.width,self.node[1] + 0.5 * self.width,self.node[2] + 0.5 * self.width])
            nodes.append([self.node[0] - 0.5 * self.width,self.node[1] + 0.5 * self.width,self.node[2] - 0.5 * self.width])
            nodes.append([self.node[0] - 0.5 * self.width,self.node[1] - 0.5 * self.width,self.node[2] + 0.5 * self.width])
            nodes.append([self.node[0] - 0.5 * self.width,self.node[1] - 0.5 * self.width,self.node[2] - 0.5 * self.width])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            i = 0
            while i < len(self.skelepts):
                if self.skelepts[i].x > self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[0].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z < self.node[2]:
                    points[1].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y < self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[2].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y < self.node[1] and self.skelepts[i].z < self.node[2]:
                    points[3].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[4].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z < self.node[2]:
                    points[5].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y < self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[6].append(self.skelepts[i])
                else:
                    points[7].append(self.skelepts[i])
                i += 1
        i = 0
        while i < len(nodes):
            self.leafs.append(SplitTree(points[i], nodes[i], self.width / 2,dim=self.dim))
            i += 1
        self.skelepts = []
        
    def addpoints(self, points : list = [],*,rads : list = [],spts : bool = False):
        #INPUTS points [skpt1,skpt2...] / [[x1,y1],[x2,y2],..]
        #Goes through all points and adds them to needed areas
        #First checks if the current leaf has been subdivided yet
        if len(points) == 0:
            return
        elif not(isinstance(points[0],SkelePoint)):
            i = 0
            tp = []
            while i < len(points):
                if len(rads) > 0:
                    tp.append(SkelePoint(points[i],rad=rads[i],connections=1))
                else:
                    tp.append(SkelePoint(points[i],connections=1))
                i += 1
            points = tp
        self.count += len(points)
        if not(self.state):#If final Leaf
            i = 0
            while i < len(points):
                self.skelepts.append(points[i])
                i += 1
            if len(self.skelepts) > self.maxpts:
                #If points more, than subdivide and add the point
                self.state = True
                self.subdivide()
        else:
            i = 0
            ne,se,nw,sw,a,b,c,d,e,f,g,h = [],[],[],[],[],[],[],[],[],[],[],[]
            while i < len(points):
                if self.dim == 2:
                    if points[i].x > self.node[0] and points[i].y > self.node[1]:
                        ne.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y < self.node[1]:
                        se.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y > self.node[1]:
                        nw.append(points[i])
                    else:
                        sw.append(points[i])
                else:
                    if points[i].x > self.node[0] and points[i].y > self.node[1] and points[i].z > self.node[2]:
                        a.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y > self.node[1] and points[i].z < self.node[2]:
                        b.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y < self.node[1] and points[i].z > self.node[2]:
                        c.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y < self.node[1] and points[i].z < self.node[2]:
                        d.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y > self.node[1] and points[i].z > self.node[2]:
                        e.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y > self.node[1] and points[i].z < self.node[2]:
                        f.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y < self.node[1] and points[i].z > self.node[2]:
                        g.append(points[i])
                    else:
                        h.append(points[i])
                i += 1
            if self.dim == 2:
                self.leafs[0].addpoints(ne)
                self.leafs[1].addpoints(se)
                self.leafs[2].addpoints(nw)
                self.leafs[3].addpoints(sw)
            else:
                self.leafs[0].addpoints(a)
                self.leafs[1].addpoints(b)
                self.leafs[2].addpoints(c)
                self.leafs[3].addpoints(d)
                self.leafs[4].addpoints(e)
                self.leafs[5].addpoints(f)
                self.leafs[6].addpoints(g)
                self.leafs[7].addpoints(h)
            
                
    def exists(self,point : list,tolerance : float,depth : int = 0):
        if self.state == True:
            #Go Deeper
            if self.dim == 2:
                if point[0] > self.node[0] and point[1] > self.node[1]:
                    ret,dep = self.leafs[0].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] < self.node[1]:
                    ret,dep = self.leafs[1].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] > self.node[1]:
                    ret,dep = self.leafs[2].exists(point,tolerance,depth + 1)
                else:
                    ret,dep = self.leafs[3].exists(point,tolerance,depth + 1)
            else:
                if point[0] > self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[0].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret,dep = self.leafs[1].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[2].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] < self.node[2]:
                    ret,dep = self.leafs[3].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[4].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret,dep = self.leafs[5].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[6].exists(point,tolerance,depth + 1)
                else:
                    ret,dep = self.leafs[7].exists(point,tolerance,depth + 1)
            if ret == False and dep - depth < 2:
                if self.dim == 2:
                    ret0,dep0 = self.leafs[0].exists(point,tolerance,depth + 1)
                    ret1,dep1 = self.leafs[1].exists(point,tolerance,depth + 1)
                    ret2,dep2 = self.leafs[2].exists(point,tolerance,depth + 1)
                    ret3,dep3 = self.leafs[3].exists(point,tolerance,depth + 1)
                    if ret0 == True:
                        dep = dep0
                        ret = True
                    elif ret1 == True:
                        dep = dep1
                        ret = True
                    elif ret2 == True:
                        dep = dep2
                        ret = True
                    elif ret3 == True:
                        dep = dep3
                        ret = True
                else:
                    ret0,dep0 = self.leafs[0].exists(point,tolerance,depth + 1)
                    ret1,dep1 = self.leafs[1].exists(point,tolerance,depth + 1)
                    ret2,dep2 = self.leafs[2].exists(point,tolerance,depth + 1)
                    ret3,dep3 = self.leafs[3].exists(point,tolerance,depth + 1)
                    ret4,dep4 = self.leafs[4].exists(point,tolerance,depth + 1)
                    ret5,dep5 = self.leafs[5].exists(point,tolerance,depth + 1)
                    ret6,dep6 = self.leafs[6].exists(point,tolerance,depth + 1)
                    ret7,dep7 = self.leafs[7].exists(point,tolerance,depth + 1)
                    if ret0 == True:
                        dep = dep0
                        ret = True
                    elif ret1 == True:
                        dep = dep1
                        ret = True
                    elif ret2 == True:
                        dep = dep2
                        ret = True
                    elif ret3 == True:
                        dep = dep3
                        ret = True
                    elif ret4 == True:
                        dep = dep4
                        ret = True
                    elif ret5 == True:
                        dep = dep5
                        ret = True
                    elif ret6 == True:
                        dep = dep6
                        ret = True
                    elif ret7 == True:
                        dep = dep7
                        ret = True
        else:
            #Search here for it. 
            i = 0
            cdis = 0
            cpoint = []
            while i < len(self.skelepts):
                #Gets closest point in container to the search
                if i == 0:
                    cpoint = self.skelepts[i].getPoint()
                    cdis = getDistance(point, cpoint)
                else:
                    tpoint = self.skelepts[i].getPoint()
                    tdis = getDistance(point, tpoint)
                    if tdis < cdis:
                        cpoint = tpoint
                        cdis = tdis
                if cdis < tolerance:
                    self.skelepts[i].addConnection()
                    return True, depth
                i += 1
            return False, depth
        return ret,dep
    
    def getConnections(self, point : list,*,getpoint = False):
        if self.state == True:
            #Go Deeper
            if self.dim == 2:
                if point[0] > self.node[0] and point[1] > self.node[1]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] < self.node[1]:
                    ret = self.leafs[1].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] > self.node[1]:
                    ret = self.leafs[2].getConnections(point,getpoint=getpoint)
                else:
                    ret = self.leafs[3].getConnections(point,getpoint=getpoint)
            else:
                if point[0] > self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] < self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                else:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
        else:
            #Search here for it. 
            i = 0
            j = 0 
            cdis = 0
            cpoint = []
            while i < len(self.skelepts):
                #Gets closest point in container to the search
                if i == 0:
                    cpoint = self.skelepts[i].getPoint()
                    cdis = getDistance(point, cpoint)
                else:
                    tpoint = self.skelepts[i].getPoint()
                    tdis = getDistance(point, tpoint)
                    if tdis < cdis:
                        cpoint = tpoint
                        cdis = tdis
                        j = i
                i += 1
            if getpoint:
                return self.skelepts[j]    
            else:
                return self.skelepts[j].connections
        return ret
    
    
    
    def plot(self,theta : list):
        if self.state == True:
            #If the figure is subdivided
            i = 0
            while i < len(self.leafs):
                self.leafs[i].plot(theta)
                i += 1
            #plt.scatter(self.node[0],self.node[1],color='purple')
        else:
            #plt.plot([self.node[0] - self.width,self.node[0] - self.width],[self.node[1] + self.width,self.node[1] - self.width],5,color='orange')
            #plt.plot([self.node[0] + self.width,self.node[0] + self.width],[self.node[1] + self.width,self.node[1] - self.width],5,color='orange')
            #plt.plot([self.node[0] + self.width,self.node[0] - self.width],[self.node[1] + self.width,self.node[1] + self.width],5,color='orange')
            #plt.plot([self.node[0] + self.width,self.node[0] - self.width],[self.node[1] - self.width,self.node[1] - self.width],5,color='orange')
            if len(self.skelepts) > 0:
                i = 0
                tx = []
                ty = []
                while i < len(self.skelepts):
                    tx.append(self.skelepts[i].x)
                    ty.append(self.skelepts[i].y)
                    plt.plot(self.skelepts[i].x + np.cos(theta) * self.skelepts[i].r,self.skelepts[i].y + np.sin(theta) * self.skelepts[i].r,5,color='red')
                    i += 1
                plt.scatter(tx,ty,5,color='green')
        
        

class SkelePoint:
#This is a class which has a point which holds x,y,z and r

    def __init__(self,point : list,*, rad : float = 0.0,connections : int = 0):
        if len(point) != 0:
            self.x = point[0]
            self.y = point[1]
            self.r = rad
            self.connections = connections
            if len(point) == 3:
                self.z = point[2]
                self.dimensions = 3
            else:
                self.dimensions = 2
            self.ordered = False
    # @profile
    def getPoint(self):
        if self.dimensions == 2:
            return [self.x,self.y]
        else:
            return [self.x,self.y,self.z]

    def getRad(self):
        return self.r

    def addConnection(self):
        self.connections += 1
    
    def getAxis(self,dim : int):
        if dim == 0:
            return self.x
        elif dim == 1:
            return self.y
        elif self.dimensions == 3:
            if dim == 2:
                return self.z
    # @profile
    def locEx(self,point):
        if point == 0:
            return False
        elif isinstance(point,list):
            if len(point) == 0:
                return False
            pt = self.getPoint()
            return pt == point
        else:
            i = 0
            while i < self.dimensions:
                if self.getAxis(i) != point.getAxis(i):
                    return False
                i += 1
        return True
    



















