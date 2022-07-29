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


def quicksort(points : list , dimension : int):
    #  **Points are Skeleton Points**
    return quicksortrunner(points,dimension,0,len(points)-1)


def quicksortrunner(points : list , dimension : int , first : int , last : int , rads : list = [],depth : int = 0):
    # print('sorting',depth)
    if first<last:
        splitpoint = partition(points,dimension,first,last)
        quicksortrunner(points, dimension, first, splitpoint - 1,depth=depth+1)
        quicksortrunner(points, dimension, splitpoint + 1,last,depth=depth+1)
        return points
            
def partition(points : list , dimension : int , first : int , last : int ):
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
    def __init__(self , points : list , depth : int = 0 , * , rads : list = [],dimensions : int = 0):
        print('making {}'.format(depth))
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
        if len(points) > 5:
            self.split = True
            self.axis = depth % self.dimensions
            points = quicksort(points,self.axis)
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
    def getNearR(self,searchPoint : list, exclude : list,*,getRads : bool = False):
        # print('searching...')
        dmin = 100000
        if self.split:
            pmin = self.node
            #If needs to go further
            if searchPoint[self.axis] <= self.node.getAxis(self.axis):
                tpmin,dmin =  self.leafR.getNearR(searchPoint, exclude)
                if tpmin != 0:
                    pmin = tpmin
                #Next checks node point and sees if its closer, overwrites if so
                if not(self.node.getPoint() == searchPoint) and not(self.node.getPoint() == exclude):
                    ndis = getDistance(searchPoint, self.node.getPoint())
                    if ndis < dmin:
                        pmin = self.node
                        dmin =  ndis
                tdis = self.node.getAxis(self.axis) - searchPoint[self.axis]
                if tdis <= dmin:
                    pmin1,dmin1 =  self.leafL.getNearR(searchPoint, exclude)
                    if not(pmin1 == 0) and not(pmin1.getPoint() == searchPoint) and not(pmin1.getPoint() == exclude):
                        if dmin1 < dmin:
                            dmin = dmin1
                            pmin = pmin1
            else:
                tpmin,dmin =  self.leafL.getNearR(searchPoint, exclude) 
                if tpmin != 0:
                    pmin = tpmin
                #Next checks node point and sees if its closer, overwrites if so
                if not(self.node.getPoint() == searchPoint) and not(self.node.getPoint() == exclude):
                    ndis = getDistance(searchPoint, self.node.getPoint())
                    if ndis < dmin:
                        pmin = self.node
                        dmin =  ndis
                tdis = searchPoint[self.axis] - self.node.getAxis(self.axis)
                if tdis <= dmin:
                    pmin1,dmin1 =  self.leafR.getNearR(searchPoint, exclude)
                    if not(pmin1 == 0) and not(pmin1.getPoint() == searchPoint) and not(pmin1.getPoint() == exclude):
                        if dmin1 < dmin and not(dmin1) == 0:
                            dmin = dmin1
                            pmin = pmin1
                            
        
        else:
            #If at the lowest
            i = 0
            while i < len(self.points):
                if i == 0:
                    if not(self.points[i].getPoint() == exclude) and not(self.points[i].getPoint() == searchPoint):
                        dmin = getDistance(searchPoint,self.points[i].getPoint())
                        pmin = self.points[i]
                else:
                   if not(self.points[i].getPoint() == exclude) and not(self.points[i].getPoint() == searchPoint):
                       tmin = getDistance(searchPoint,self.points[i].getPoint())
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
        
    def getVectorR(self,point : list,vec : list,n : int,depth : int = 0,*,getRads : bool = False,scan = np.pi / 4):
            #Get vector will get the closest n number of points to the search point
            #it consideres a 'scan' degree area along the given vector, will go deepest first as thats where the closest points should be
            #however it will store node value and compare on the way out if a given node is a better point than something returned
            if depth == 0:
                self.l = 0
            retPoints = []
            retDist = []
            axis = depth % self.dimensions
            node = self.node
            if getRads:
                retR = []
            if not(self.split):
                i = 0
                while i < len(self.points):
                    #each point it first checks the angle between the vectors.
                    j = 0
                    cvec = []
                    tpoint = []
                    while j < len(vec):
                        cvec.append(self.points[i].getAxis(j) - point[j])
                        tpoint.append(point[j] + vec[j])
                        j += 1
                    tdis = getDistance(point,self.points[i].getPoint())
                    theta = getAngle(vec,cvec,getDistance(point,tpoint),tdis)
                    if theta  < (scan / 2):
                        #Within vector distance
                        if len(retPoints) < n:
                            self.l += 1
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
                #First want to see if the vector only reaches a specific leaf of the tree
                i = 0
                vecax = []
                tp1 = []
                tp2 = []
                nvec = []
                while i < len(point):
                    if not(i == axis):
                        vecax.append(1)
                    else:
                        vecax.append(0)
                    tp1.append(point[i] + vec[i])
                    tp2.append(point[i] + vecax[i])
                    nvec.append(node[i] - point[i])
                    i += 1
                theta = getAngle(vec,vecax,getDistance(point,tp1),getDistance(point,tp2))
                mat = []
                if self.dimensions == 2:
                    mat.append([])
                    mat[0].append(vec[0] * np.cos(scan / 2) - vec[1] * np.sin(scan / 2))
                    mat[0].append(vec[0] * np.cos(-scan / 2) - vec[1] * np.sin(-scan / 2))
                    mat.append([])
                    mat[1].append(vec[0] * np.sin(scan / 2) + vec[1] * np.cos(scan / 2))
                    mat[1].append(vec[0] * np.sin(-scan / 2) +vec[1] * np.cos(-scan / 2))
                else:
                    if axis == 0:
                        #X axis rotation
                        mat.append([])
                        mat[0].append(vec[0] * np.cos(scan / 2) - vec[1] * np.sin(scan / 2))
                        mat[0].append(vec[0] * np.cos(-scan / 2) - vec[1]  * np.sin(-scan / 2))
                        mat.append([])
                        mat[1].append(vec[0] * np.sin(scan / 2) + vec[1] * np.cos(scan / 2))
                        mat[1].append(vec[0] * np.sin(-scan / 2) + vec[1] * np.cos(-scan / 2))
                        mat.append([])
                        mat[2].append(vec[2])
                        mat[2].append(vec[2])
                    elif axis == 1:
                        #Y axis rotation
                        mat.append([])
                        mat[0].append(vec[0] * np.cos(scan / 2) + vec[2] * np.sin(scan / 2))
                        mat[0].append(vec[0] * np.cos(-scan / 2) + vec[2]  * np.sin(-scan / 2))
                        mat.append([])
                        mat[1].append(vec[1])
                        mat[1].append(vec[1])
                        mat.append([])
                        mat[2].append(-vec[0] * np.sin(scan / 2) + vec[2] * np.cos(scan / 2))
                        mat[2].append(-vec[0] * np.sin(-scan / 2) + vec[2] * np.cos(-scan / 2))
                    else:
                        #Z axis rotation
                        mat.append([])
                        mat[0].append(vec[0])
                        mat[0].append(vec[0])
                        mat.append([])
                        mat[1].append(vec[1] * np.cos(scan / 2) - vec[2] * np.sin(scan / 2))
                        mat[1].append(vec[1] * np.cos(-scan / 2) - vec[2] * np.sin(-scan / 2))
                        mat.append([])
                        mat[2].append(vec[1] * np.sin(scan / 2) + vec[2] * np.cos(scan / 2))
                        mat[2].append(vec[1] * np.sin(-scan / 2) + vec[2] * np.cos(-scan / 2))
                if node.getAxis(axis) < point[axis] and theta < max(mat[axis]) and theta < min(mat[axis]):
                    retPoints,retDist = self.leafR.getVectorR(point, vec, n, depth + 1, scan = scan)
                elif node.getAxis(axis) > point[axis] and theta < max(mat[axis]) and theta < min(mat[axis]):
                    #point on left side of axis and positive vector
                    retPoints,retDist = self.leafL.getVectorR(point, vec, n, depth + 1, scan = scan)
                else:
                    tretPoints = []
                    tretDist = []
                    retPointsl = []
                    retPointsr = []
                    retDistl = []
                    retDistr = []
                    retPointsl,retDistl = self.leafL.getVectorR(point, vec, n,depth + 1, scan = scan)
                    retPointsr,retDistr = self.leafR.getVectorR(point, vec, n,depth + 1, scan = scan)
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
                    ntheta = getAngle(vec,nvec,getDistance(point,tp1),getDistance(point,node.getPoint()))
                    #gets the node point if it falls inside the criteria
                    if ntheta < (scan / 2): 
                        self.l += 1
                        tretPoints.append(node)
                        tretDist.append(getDistance(point,node.getPoint()))
                    #Finally aquires the best n# of points from big list
                    i = 0
                    while i < len(tretPoints):
                        if i < n:
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
                if getRads:
                    return retPoints,retDist, retR
                else:
                    return retPoints,retDist
        
# class kdTree:
#     def __init__(self,points : list, dim : int, * ,  rads : list = []):
        
#     #for sorting points along an axis
#         self.PointsList = []
#         self.RadsList = []
#         i = 0
#         while i < len(points):
#             self.PointsList.append(points[i])
#             if len(rads) > 0:
#                 self.RadsList.append(rads[i])
#             i = i + 1
#         self.dimensions = dim
#         if len(self.RadsList) > 0:
#             #Passes rads, any value you would want returned with tree/point
#             self.tree,self.rad = self.makeTree(self.PointsList,0,self.RadsList)
#         else:
#             self.tree = self.makeTree(self.PointsList,0)
            
#     # def sort(self,points : list, dimension,rads = []) -> list:
#         #Normal Sort, SLOW : -> Quicksort
#         # for i in range(len(points)):
#         #     print(i)
#         #     min_i = i
#         #     for j in range(i+1,len(points)):
#         #         if points[j][dimension] < points[min_i][dimension]:
#         #             min_i = j
#         #     points[i],points[min_i] = points[min_i],points[i]
#         #     if len(rads) > 0:
#         #         rads[i],rads[min_i] = rads[min_i],rads[i]
#         # if len(rads) > 0:
#         #     return points,rads    
#         # else:
#         #     return points
        
        
    
#     #contructs the tree    
#     def makeTree(self,points:list, depth : int,rads : list = []):
#         if depth == 0:
#             st = time.time()
#             print(depth,'Initiating k-d,Tree')
#         finTree = []
#         getRads = False
#         if len(rads) > 0:
#             finRadT = []
#             getRads = True
#         #designed to never have empty leafs
#         if len(points) > 5:
#             axis = depth % self.dimensions #gets axis to divide along
#             if getRads:
#                 points,rads = quicksort(points,axis,rads)    
#             else:
#                 points = quicksort(points,axis)
               
            
#             mid = len(points) // 2
#             finTree.append(points[mid])#choses node point
#             if getRads:
#                 finRadT.append(rads[mid])
#                 radsl = []
#                 radsr = []
#             pointsl = []
#             i = 0
#             while i < mid:
#                 pointsl.append(points[i])
#                 if getRads:
#                     radsl.append(rads[i])
#                 i = i + 1
#             if getRads:
#                 tempt,tempr = self.makeTree(pointsl, depth + 1,radsl)
#                 finTree.append(tempt)
#                 finRadT.append(tempr)
#             else:
#                 finTree.append(self.makeTree(pointsl,depth+1))#gets roots(Left of node)
            
#             pointsr = []
#             i = mid + 1
            
#             while i < len(points):
#                 pointsr.append(points[i])
#                 if getRads:
#                     radsr.append(rads[i])
#                 i = i + 1
#             if getRads:
#                 tempt,tempr = self.makeTree(pointsr, depth + 1,radsr)
#                 finTree.append(tempt)
#                 finRadT.append(tempr)
#             else:
#                 finTree.append(self.makeTree(pointsr,depth+1))#(right of node)
#             if depth == 0:
#                 et = time.time()
#                 tt = et - st
#                 print('k-d tree took {} minuites and {} seconds to make'.format(tt // 60,tt % 60))
#         else:
#             #add in all points, if small amount of points or too deep
#             i = 0
#             finTree.append('Full')#marks a bottom layer
#             if getRads:
#                 finRadT.append('Full')
#             while i < len(points):
#                 finTree.append(points[i])
#                 if getRads:
#                     finRadT.append(rads[i])
#                 i = i + 1
        
#         if not getRads:   
#             return finTree
#         else:
#             return finTree , finRadT
      
#     #searches tree              
#     def getNearR(self,searchPoint : list , exclude : list ,tree : list = [], depth : int = 0,*,getRads : bool = False, rtree : list = []):
#         smallestLayer = []
#         if getRads:
#             smallestr = []
#         dmin = 0
#         if depth == 0:
#             tree = self.tree
#             if getRads:
#                 rtree = self.rad
#         node = tree[0]
#         if getRads:
#             rnode = rtree[0]
#         axis = depth % self.dimensions
#         if node == 'Full':
#             #if searching on a bottom Layer, will brute force
#             i = 1
#             dmin = 100000000000
#             while i < len(tree):
#                 if(tree[i] != exclude and tree[i] != searchPoint):
#                     if i == 1:
#                         dmin = getDistance(searchPoint, tree[1])
#                         smallestLayer = tree[1]
#                         if getRads:
#                             smallestr = rtree[1]
#                     dtemp = getDistance(searchPoint,tree[i])
#                     if dtemp < dmin and dtemp != 0:
#                         dmin = dtemp
#                         smallestLayer = tree[i]
#                         if getRads:
#                             smallestr = tree[i]
#                 i = i + 1
#         else:
#             if searchPoint[axis] <= tree[0][axis]:
#                 #want to go deeper to the left
#                 if getRads:
#                     smallestLayer,smallestr = self.getNearR(searchPoint,exclude,tree[1],depth + 1,getRads=True,rtree=rtree[1])
#                 else:
#                     smallestLayer = self.getNearR(searchPoint,exclude,tree[1], depth + 1)
#             else:
#                 #want to go deeper to the right
#                 if getRads:
#                     smallestLayer,smallestr = self.getNearR(searchPoint,exclude,tree[2],depth + 1,getRads=True,rtree=rtree[2])
#                 else:
#                     smallestLayer = self.getNearR(searchPoint,exclude,tree[2], depth + 1)
#             #gets the current closest points distance before checks
#             if smallestLayer == None or smallestLayer == []:
#                 tdis = 10000000000
#             else:
#                 tdis = getDistance(searchPoint,smallestLayer)
            
#             if tree[0] != exclude and tree[0] != searchPoint:
#                 #checks if current node is closer than found
#                 tdis1 = getDistance(searchPoint,node)
#                 if tdis1 < tdis and tdis1 != 0:
#                     smallestLayer = node
#                     if getRads:
#                         smallestr = rnode
#                     tdis = tdis1
                
#             #checks if the circle crosses the plane and needs to check other leafs on the other side
#             #reduces some of the math/ steps needed because i made it find the axis distance from the search 
#             #point to the node. so if theres a possibility there could be a point itll check it but have to do less math
#             #and back and fourth sending to make it happen
#             tPoint = None
#             if searchPoint[axis] <= tree[0][axis]:
#                 tdis3 = node[axis] - searchPoint[axis]
#                 if tdis3 <= tdis and tdis3 != 0:
#                     if getRads:
#                         tPoint,tRad = self.getNearR(searchPoint, exclude,tree[2],depth + 1,getRads=True,rtree=rtree[2])    
#                     else:
#                         tPoint = self.getNearR(searchPoint,exclude,tree[2], depth + 1)
#             else:
#                 tdis3 = searchPoint[axis] - node[axis]
#                 if tdis3 <= tdis and tdis3 != 0:
#                     if getRads:
#                         tPoint,tRad = self.getNearR(searchPoint, exclude,tree[1],depth + 1,getRads=True,rtree=rtree[1])    
#                     else:
#                         tPoint = self.getNearR(searchPoint,exclude,tree[1], depth + 1)
#             if tPoint != None and tPoint != []:
#                 tdis2 = getDistance(searchPoint,tPoint)
#                 if tdis2 < tdis and tdis2 != 0:
#                     smallestLayer = tPoint
#                     if getRads:
#                         smallestr = tRad
#                     tdis = tdis2
#         if getRads:
#             return smallestLayer,smallestr
#         else:
#             return smallestLayer
                    
                
#     def getInR(self,point : list, dim : float, mode : int,tree : list = [],depth : int = 0,*,getRads : bool = False,rtree : list = []):
#         #Returns all the points which lie inside a give area arround a certain point
#         #Mode 0 => Square area, point in center, side = 2 * dim
#         #Mode 1 => Circle area, point in center, rad  = dim
#         retPoints = []
#         axis = depth % self.dimensions
#         if depth == 0:
#             tree = self.tree
#             if getRads:
#                 rtree = self.rad
#         if getRads:
#             retR = []
#             rnode = rtree[0]
#         node = tree[0]        
#         if node == 'Full':
#             i = 1
#             while i < len(tree):
#                 if mode == 0:
#                     if tree[i][0] >= point[0] - dim and tree[i][0] <= point[0] + dim:
#                         if tree[i][1] >= point[1] - dim and tree[i][1] <= point[1] + dim:
#                             if self.dmensions == 2 or (self.dimensions == 3 and tree[i][2] >= point[2] - dim and tree[i][2] <= point[2] + dim):
#                                 retPoints.append(tree[i])
#                                 if getRads:
#                                     retR.append(rtree[i])
#                 else:
#                     if getDistance(point,tree[i]) <= dim:
#                         retPoints.append(tree[i])
#                         if getRads:
#                             retR.append(rtree[i])
#                 i += 1
#         else:
#             #Uses square to obtain all possible sections that might be needed
#             #mode only comes into play when searching at bottom layers, however adding the node points 
#             #will still depend on mode  
#             pts = []
#             rads = []
#             if point[axis] - dim > node[axis]:
#                 if getRads:
#                     pts,rads = self.getInR(point,dim,mode,tree[2],depth + 1, getRads=True,rtree = rtree[2])
#                 else:
#                     pts = self.getInR(point,dim,mode,tree[2],depth + 1)
#             elif point[axis] + dim < node[axis]:
#                 if getRads:
#                     pts,rads = self.getInR(point,dim,mode,tree[1],depth + 1, getRads = True, rtree = rtree[1])
#                 else:
#                     pts = self.getInR(point,dim,mode,tree[1],depth + 1)
#             else:
#                 if getRads:
#                     pts1,rad1 = self.getInR(point,dim,mode,tree[1],depth + 1, getRads = True, rtree = rtree[1])
#                     pts2,rad2 = self.getInR(point,dim,mode,tree[2],depth + 1, getRads = True, rtree = rtree[2])
#                 else:
#                     pts1 = self.getInR(point,dim,mode,tree[1],depth + 1)
#                     pts2 = self.getInR(point,dim,mode,tree[2],depth + 1)
#                 i = 0
#                 while i < len(pts1):
#                     pts.append(pts1[i])
#                     if getRads:
#                         rads.append(rad1[i])
#                     i += 1
#                 i = 0
#                 while i < len(pts2):
#                     pts.append(pts2[i])
#                     if getRads:
#                         rads.append(rad2[i])
#                     i += 1
#                 #Note only needs to check node here
#                 if mode == 0:
#                     if node[0] >= point[0] - dim and node[0] <= point[0] + dim:
#                         if node[1] >= point[1] - dim and node[1] <= point[1] + dim:
#                             if self.dmensions == 2 or (self.dimensions == 3 and node[2] >= point[2] - dim and node[2] <= point[2] + dim):
#                                 pts.append(node)
#                                 if getRads:
#                                     rads.append(rnode)
#                 else:
#                     if getDistance(point,node) <= dim:
#                         pts.append(node)
#                         if getRads:
#                             rads.append(rnode)
                
#             if len(pts) > 0:
#                 i = 0
#                 while i < len(pts):
#                     retPoints.append(pts[i])
#                     if getRads:
#                         retR.append(rads[i])
#                     i += 1
#         if getRads:
#             return retPoints,retR
#         else:
#             return retPoints
    
#     def getVectorR(self,point : list,vec : list,n : int,tree : list = [],depth : int = 0,*,getRads : bool = False,rtree : list = [],scan = np.pi / 4):
#         #Get vector will get the closest n number of points to the search point
#         #it consideres a 'scan' degree area along the given vector, will go deepest first as thats where the closest points should be
#         #however it will store node value and compare on the way out if a given node is a better point than something returned
#         if depth == 0:
#             self.l = 0
        
#         retPoints = []
#         retDist = []
#         axis = depth % self.dimensions
#         if depth == 0:
#             tree = self.tree
#             if getRads:
#                 rtree = self.rad
#         node =  tree[0]
#         if getRads:
#             rnode = rtree[0]
#             retR = []
#         if node == 'Full':
#             i = 1
#             while i < len(tree):
#                 #each point it first checks the angle between the vectors.
#                 j = 0
#                 cvec = []
#                 tpoint = []
#                 while j < len(vec):
#                     cvec.append(tree[i][j] - point[j])
#                     tpoint.append(point[j] + vec[j])
#                     j += 1
#                 tdis = getDistance(point, tree[i])
#                 theta = getAngle(vec,cvec,getDistance(point,tpoint),tdis)
#                 if theta  < (scan / 2):
#                     #Within vector distance
#                     if len(retPoints) < n:
#                         self.l += 1
#                         retPoints.append(tree[i])
#                         retDist.append(tdis)
#                         if getRads:
#                             retR.append(rtree[i])
#                     else:
#                         j = 0
#                         tagj = 0
#                         mtdis = -1
#                         while j < len(retPoints):
#                             if tdis < retDist[j] and retDist[j] > mtdis:
#                                 tagj = j
#                                 mtdis = retDist[j]
#                             j += 1
#                         if not(mtdis == -1):
#                             retPoints[tagj] = tree[i]
#                             retDist[tagj] = tdis 
#                             if getRads:
#                                 retR[tagj] = rtree[i]
#                 i += 1
#         else:
#             #First want to see if the vector only reaches a specific leaf of the tree
#             i = 0
#             vecax = []
#             tp1 = []
#             tp2 = []
#             nvec = []
#             while i < len(point):
#                 if not(i == axis):
#                     vecax.append(1)
#                 else:
#                     vecax.append(0)
#                 tp1.append(point[i] + vec[i])
#                 tp2.append(point[i] + vecax[i])
#                 nvec.append(node[i] - point[i])
#                 i += 1
#             theta = getAngle(vec,vecax,getDistance(point,tp1),getDistance(point,tp2))
#             mat = []
#             if self.dimensions == 2:
#                 mat.append([])
#                 mat[0].append(vec[0] * np.cos(scan / 2) - vec[1] * np.sin(scan / 2))
#                 mat[0].append(vec[0] * np.cos(-scan / 2) - vec[1] * np.sin(-scan / 2))
#                 mat.append([])
#                 mat[1].append(vec[0] * np.sin(scan / 2) + vec[1] * np.cos(scan / 2))
#                 mat[1].append(vec[0] * np.sin(-scan / 2) +vec[1] * np.cos(-scan / 2))
#             else:
#                 if axis == 0:
#                     #X axis rotation
#                     mat.append([])
#                     mat[0].append(vec[0] * np.cos(scan / 2) - vec[1] * np.sin(scan / 2))
#                     mat[0].append(vec[0] * np.cos(-scan / 2) - vec[1]  * np.sin(-scan / 2))
#                     mat.append([])
#                     mat[1].append(vec[0] * np.sin(scan / 2) + vec[1] * np.cos(scan / 2))
#                     mat[1].append(vec[0] * np.sin(-scan / 2) + vec[1] * np.cos(-scan / 2))
#                     mat.append([])
#                     mat[2].append(vec[2])
#                     mat[2].append(vec[2])
#                 elif axis == 1:
#                     #Y axis rotation
#                     mat.append([])
#                     mat[0].append(vec[0] * np.cos(scan / 2) + vec[2] * np.sin(scan / 2))
#                     mat[0].append(vec[0] * np.cos(-scan / 2) + vec[2]  * np.sin(-scan / 2))
#                     mat.append([])
#                     mat[1].append(vec[1])
#                     mat[1].append(vec[1])
#                     mat.append([])
#                     mat[2].append(-vec[0] * np.sin(scan / 2) + vec[2] * np.cos(scan / 2))
#                     mat[2].append(-vec[0] * np.sin(-scan / 2) + vec[2] * np.cos(-scan / 2))
#                 else:
#                     #Z axis rotation
#                     mat.append([])
#                     mat[0].append(vec[0])
#                     mat[0].append(vec[0])
#                     mat.append([])
#                     mat[1].append(vec[1] * np.cos(scan / 2) - vec[2] * np.sin(scan / 2))
#                     mat[1].append(vec[1] * np.cos(-scan / 2) - vec[2] * np.sin(-scan / 2))
#                     mat.append([])
#                     mat[2].append(vec[1] * np.sin(scan / 2) + vec[2] * np.cos(scan / 2))
#                     mat[2].append(vec[1] * np.sin(-scan / 2) + vec[2] * np.cos(-scan / 2))
#             if node[axis] < point[axis] and theta < max(mat[axis]) and theta < min(mat[axis]) :
#                 #point on right side of axis and positive vector
#                 if getRads:
#                     retPoints,retDist,retR = self.getVectorR(point, vec, n, tree[2], depth + 1, getRads = True, rtree = rtree[2], scan = scan)
#                 else:
#                     retPoints,retDist = self.getVectorR(point, vec, n, tree[2], depth + 1, scan = scan)
#             elif node[axis] > point[axis] and theta < max(mat[axis]) and theta < min(mat[axis]):
#                 #point on left side of axis and positive vector
#                 if getRads:
#                     retPoints,retDist,retR = self.getVectorR(point, vec, n, tree[1], depth + 1, getRads = True, rtree = rtree[1], scan = scan)
#                 else:
#                     retPoints,retDist = self.getVectorR(point, vec, n, tree[1], depth + 1, scan = scan)
#             else:
#                 tretPoints = []
#                 tretDist = []
#                 retPointsl = []
#                 retPointsr = []
#                 retDistl = []
#                 retDistr = []
#                 if getRads:
#                     retRl = []
#                     retRr = []
#                     tretR = []
#                     retPointsl,retDistl,retRl = self.getVectorR(point, vec, n,tree[1],depth + 1, getRads = True,rtree = rtree[1], scan = scan)
#                     retPointsr,retDistr,retRr = self.getVectorR(point, vec, n,tree[2],depth + 1, getRads = True,rtree = rtree[2], scan = scan)
#                 else:
#                     retPointsl,retDistl = self.getVectorR(point, vec, n,tree[1],depth + 1, scan = scan)
#                     retPointsr,retDistr = self.getVectorR(point, vec, n,tree[2],depth + 1, scan = scan)
                
#                 i = 0
#                 while i < len(retPointsl):
#                     tretPoints.append(retPointsl[i])
#                     tretDist.append(retDistl[i])
#                     if getRads:
#                         tretR.append(retRl[i])
#                     i += 1
#                 i = 0
#                 while i < len(retPointsr):
#                     tretPoints.append(retPointsr[i])
#                     tretDist.append(retDistr[i])
#                     if getRads:
#                         tretR.append(retRr[i])
#                     i += 1
#                 ntheta = getAngle(vec,nvec,getDistance(point,tp1),getDistance(point,node))
#                 #gets the node point if it falls inside the criteria
#                 if ntheta < (scan / 2): 
#                     self.l += 1
#                     tretPoints.append(node)
#                     tretDist.append(getDistance(point,node))
#                     if getRads:     
#                         tretR.append(rnode)
#                 #Finally aquires the best n# of points from big list
#                 i = 0
#                 while i < len(tretPoints):
#                     if i < n:
#                         retPoints.append(tretPoints[i])
#                         retDist.append(tretDist[i])
#                         if getRads:
#                             retR.append(tretR[i])
#                     else:
#                         tdis = tretDist[i]
#                         j = 0
#                         tagj = 0
#                         mtdis = -1
#                         while j < len(retPoints):
                            
#                             if tdis < retDist[j] and retDist[j] > mtdis:
#                                 tagj = j
#                                 mtdis = retDist[j]
#                             j += 1
#                         if not(mtdis == -1):
#                             retPoints[tagj] = tretPoints[i]
#                             retDist[tagj] = tdis 
#                             if getRads:
#                                 retR[tagj] = tretR[i]
#                     i += 1
                
#         if depth == 0:
#             if getRads:
#                 return retPoints, retR
#             else:
#                 return retPoints
#         else:
#             if getRads:
#                 return retPoints,retDist, retR
#             else:
#                 return retPoints,retDist
#     def treeLines(self,bounds : list,onode : list = [],side : int = 0,tree : list = [],depth : int = 0) -> list:
#         #Works for 2D 
#         #input: bounds format [[top left] , [bottom right]], both with [x,y]
#         #output: [ [a1x,a1y] , [a2x,a2y] , [b1x,b1y] , [b2x,b2y] , ...] coupled points
#         ret = []
#         axis = depth % self.dimensions
#         if depth == 0:
#             tree = self.tree
#             onode = [0,bounds[1][1]]
#         node = tree[0]
#         if axis == 0:#drawing straight along the y
#             if len(tree) == 3 and node != 'Full':
#                 if side == 0:
#                     ret.append([node[0],onode[1]])
#                     ret.append([node[0],bounds[0][1]])
#                 else:
#                     ret.append([node[0],onode[1]])
#                     ret.append([node[0],bounds[1][1]])
#                 boundsleft = [bounds[0],[node[0],bounds[1][1]]]
#                 left = self.treeLines(boundsleft,node,1,tree[1],depth + 1)
#                 if left != None:
#                     i = 0
#                     while i < len(left):
#                         ret.append([left[i][0],left[i][1]])
#                         i = i + 1
#                 boundsright = [[node[0],bounds[0][1]],bounds[1]]
#                 right = self.treeLines(boundsright,node,0,tree[2],depth + 1)
#                 if right != None:
#                     i = 0 
#                     while i < len(right):
#                         ret.append([right[i][0],right[i][1]])
#                         i = i + 1
#         else:#drwaing straight along the x
            
#             if len(tree) == 3 and node != 'Full':
#                 if side == 0:
#                     ret.append([onode[0],node[1]])
#                     ret.append([bounds[1][0],node[1]])
#                 else:
#                     ret.append([onode[0],node[1]])
#                     ret.append([bounds[0][0],node[1]])
#                 boundsleft = [[bounds[0][0],node[1]],bounds[1]]
#                 left = self.treeLines(boundsleft,node,1,tree[1],depth + 1)
#                 i = 0
#                 if left != None:
#                     while i < len(left):
#                         ret.append([left[i][0],left[i][1]])
#                         i = i + 1
#                 boundsright = [bounds[0],[bounds[1][0],node[0]]]
#                 right = self.treeLines(boundsright,node,0,tree[2],depth + 1)
#                 if right != None:
#                     i = 0
#                     while i < len(right):
#                         ret.append([right[i][0],right[i][1]])
#                         i = i + 1
#         if ret != []:
#             return ret



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
    



















