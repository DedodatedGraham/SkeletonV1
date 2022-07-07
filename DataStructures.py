# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:46:08 2022

@author: graha
"""
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

import numpy as np

from Skeletize import getDistance, getAngle

class kdTree:
    
    def __init__(self,points : list, dim : int, * ,  rads : list = []):
        
    #for sorting points along an axis
        self.PointsList = []
        i = 0
        while i < len(points):
            self.PointsList.append(points[i])
            i = i + 1
        self.dimensions = dim
        if len(rads) > 0:
            #Passes rads, any value you would want returned with tree/point
            self.tree,self.rad = self.makeTree(self.PointsList,0,rads)
        else:
            self.tree = self.makeTree(self.PointsList,0)
            
    def sort(self,points : list, dimension,rads = []) -> list:
        for i in range(len(points)):
            min_i = i
            for j in range(i+1,len(points)):
                if points[j][dimension] < points[min_i][dimension]:
                    min_i = j
            points[i],points[min_i] = points[min_i],points[i]
            if len(rads) > 0:
                rads[i],rads[min_i] = rads[min_i],rads[i]
        if len(rads) > 0:
            return points,rads    
        else:
            return points
    
    #contructs the tree    
    def makeTree(self,points:list, depth : int,rads : list = []):
        finTree = []
        getRads = False
        if len(rads) > 0:
            finRadT = []
            getRads = True
        #designed to never have empty leafs
        if len(points) > 5 and depth < 100:
            axis = depth % self.dimensions #gets axis to divide along
            if getRads:
                points,rads = self.sort(points,axis,rads)    
            else:
                points = self.sort(points,axis)
               
            
            mid = len(points) // 2
            finTree.append(points[mid])#choses node point
            if getRads:
                finRadT.append(rads[mid])
                radsl = []
                radsr = []
            pointsl = []
            i = 0
            while i < mid:
                pointsl.append(points[i])
                if getRads:
                    radsl.append(rads[i])
                i = i + 1
            if getRads:
                tempt,tempr = self.makeTree(pointsl, depth + 1,radsl)
                finTree.append(tempt)
                finRadT.append(tempr)
            else:
                finTree.append(self.makeTree(pointsl,depth+1))#gets roots(Left of node)
            
            pointsr = []
            i = mid + 1
            
            while i < len(points):
                pointsr.append(points[i])
                if getRads:
                    radsr.append(rads[i])
                i = i + 1
            if getRads:
                tempt,tempr = self.makeTree(pointsr, depth + 1,radsr)
                finTree.append(tempt)
                finRadT.append(tempr)
            else:
                finTree.append(self.makeTree(pointsr,depth+1))#(right of node)
            
        else:
            #add in all points, if small amount of points or too deep
            i = 0
            finTree.append('Full')#marks a bottom layer
            if getRads:
                finRadT.append('Full')
            while i < len(points):
                finTree.append(points[i])
                if getRads:
                    finRadT.append(rads[i])
                i = i + 1
        
        if not getRads:   
            return finTree
        else:
            return finTree , finRadT
      
    #searches tree              
    def getNearR(self,searchPoint : list , exclude : list ,tree : list = [], depth : int = 0,*,getRads : bool = False, rtree : list = []):
        smallestLayer = []
        if getRads:
            smallestr = []
        dmin = 0
        if depth == 0:
            tree = self.tree
            if getRads:
                rtree = self.rad
        node = tree[0]
        if getRads:
            rnode = rtree[0]
        axis = depth % self.dimensions
        if node == 'Full':
            #if searching on a bottom Layer, will brute force
            i = 1
            dmin = 100000000000
            while i < len(tree):
                if(tree[i] != exclude and tree[i] != searchPoint):
                    if i == 1:
                        dmin = getDistance(searchPoint, tree[1])
                        smallestLayer = tree[1]
                        if getRads:
                            smallestr = rtree[1]
                    dtemp = getDistance(searchPoint,tree[i])
                    if dtemp < dmin and dtemp != 0:
                        dmin = dtemp
                        smallestLayer = tree[i]
                        if getRads:
                            smallestr = tree[i]
                i = i + 1
        else:
            if searchPoint[axis] <= tree[0][axis]:
                #want to go deeper to the left
                if getRads:
                    smallestLayer,smallestr = self.getNearR(searchPoint,exclude,tree[1],depth + 1,getRads=True,rtree=rtree[1])
                else:
                    smallestLayer = self.getNearR(searchPoint,exclude,tree[1], depth + 1)
            else:
                #want to go deeper to the right
                if getRads:
                    smallestLayer,smallestr = self.getNearR(searchPoint,exclude,tree[2],depth + 1,getRads=True,rtree=rtree[2])
                else:
                    smallestLayer = self.getNearR(searchPoint,exclude,tree[2], depth + 1)
            #gets the current closest points distance before checks
            if smallestLayer == None or smallestLayer == []:
                tdis = 10000000000
            else:
                tdis = getDistance(searchPoint,smallestLayer)
            
            if tree[0] != exclude and tree[0] != searchPoint:
                #checks if current node is closer than found
                tdis1 = getDistance(searchPoint,node)
                if tdis1 < tdis and tdis1 != 0:
                    smallestLayer = node
                    if getRads:
                        smallestr = rnode
                    tdis = tdis1
                
            #checks if the circle crosses the plane and needs to check other leafs on the other side
            #reduces some of the math/ steps needed because i made it find the axis distance from the search 
            #point to the node. so if theres a possibility there could be a point itll check it but have to do less math
            #and back and fourth sending to make it happen
            tPoint = None
            if searchPoint[axis] <= tree[0][axis]:
                tdis3 = node[axis] - searchPoint[axis]
                if tdis3 <= tdis and tdis3 != 0:
                    if getRads:
                        tPoint,tRad = self.getNearR(searchPoint, exclude,tree[2],depth + 1,getRads=True,rtree=rtree[2])    
                    else:
                        tPoint = self.getNearR(searchPoint,exclude,tree[2], depth + 1)
            else:
                tdis3 = searchPoint[axis] - node[axis]
                if tdis3 <= tdis and tdis3 != 0:
                    if getRads:
                        tPoint,tRad = self.getNearR(searchPoint, exclude,tree[1],depth + 1,getRads=True,rtree=rtree[1])    
                    else:
                        tPoint = self.getNearR(searchPoint,exclude,tree[1], depth + 1)
            if tPoint != None and tPoint != []:
                tdis2 = getDistance(searchPoint,tPoint)
                if tdis2 < tdis and tdis2 != 0:
                    smallestLayer = tPoint
                    if getRads:
                        smallestr = tRad
                    tdis = tdis2
        if getRads:
            return smallestLayer,smallestr
        else:
            return smallestLayer
                    
                
    def getInR(self,point : list, dim : float, mode : int,tree : list = [],depth : int = 0,*,getRads : bool = False,rtree : list = []):
        #Returns all the points which lie inside a give area arround a certain point
        #Mode 0 => Square area, point in center, side = 2 * dim
        #Mode 1 => Circle area, point in center, rad  = dim
        retPoints = []
        axis = depth % self.dimensions
        if depth == 0:
            tree = self.tree
            if getRads:
                rtree = self.rad
        if getRads:
            retR = []
            rnode = rtree[0]
        node = tree[0]        
        if node == 'Full':
            i = 1
            while i < len(tree):
                if mode == 0:
                    if tree[i][0] >= point[0] - dim and tree[i][0] <= point[0] + dim:
                        if tree[i][1] >= point[1] - dim and tree[i][1] <= point[1] + dim:
                            if self.dmensions == 2 or (self.dimensions == 3 and tree[i][2] >= point[2] - dim and tree[i][2] <= point[2] + dim):
                                retPoints.append(tree[i])
                                if getRads:
                                    retR.append(rtree[i])
                else:
                    if getDistance(point,tree[i]) <= dim:
                        retPoints.append(tree[i])
                        if getRads:
                            retR.append(rtree[i])
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

    
    def getVectorR(self,point : list,vec : list,n : int,tree : list = [],depth : int = 0,*,getRads : bool = False,rtree : list = [],scan = np.pi / 4):
        #Get vector will get the closest n number of points to the search point
        #it consideres a 'scan' degree area along the given vector, will go deepest first as thats where the closest points should be
        #however it will store node value and compare on the way out if a given node is a better point than something returned
        if depth == 0:
            self.l = 0
        
        retPoints = []
        retDist = []
        axis = depth % self.dimensions
        if depth == 0:
            tree = self.tree
            if getRads:
                rtree = self.rad
        node =  tree[0]
        if getRads:
            rnode = rtree[0]
            retR = []
        if node == 'Full':
            i = 1
            while i < len(tree):
                #each point it first checks the angle between the vectors.
                j = 0
                cvec = []
                tpoint = []
                while j < len(vec):
                    cvec.append(tree[i][j] - point[j])
                    tpoint.append(point[j] + vec[j])
                    j += 1
                tdis = getDistance(point, tree[i])
                theta = getAngle(vec,cvec,getDistance(point,tpoint),tdis)
                if theta  < (scan / 2):
                    #Within vector distance
                    if len(retPoints) < n:
                        self.l += 1
                        retPoints.append(tree[i])
                        retDist.append(tdis)
                        if getRads:
                            retR.append(rtree[i])
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
                            retPoints[tagj] = tree[i]
                            retDist[tagj] = tdis 
                            if getRads:
                                retR[tagj] = rtree[i]
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
            if node[axis] < point[axis] and theta < max(mat[axis]) and theta < min(mat[axis]) :
                #point on right side of axis and positive vector
                if getRads:
                    retPoints,retDist,retR = self.getVectorR(point, vec, n, tree[2], depth + 1, getRads = True, rtree = rtree[2], scan = scan)
                else:
                    retPoints,retDist = self.getVectorR(point, vec, n, tree[2], depth + 1, scan = scan)
            elif node[axis] > point[axis] and theta < max(mat[axis]) and theta < min(mat[axis]):
                #point on left side of axis and positive vector
                if getRads:
                    retPoints,retDist,retR = self.getVectorR(point, vec, n, tree[1], depth + 1, getRads = True, rtree = rtree[1], scan = scan)
                else:
                    retPoints,retDist = self.getVectorR(point, vec, n, tree[1], depth + 1, scan = scan)
            else:
                tretPoints = []
                tretDist = []
                retPointsl = []
                retPointsr = []
                retDistl = []
                retDistr = []
                if getRads:
                    retRl = []
                    retRr = []
                    tretR = []
                    retPointsl,retDistl,retRl = self.getVectorR(point, vec, n,tree[1],depth + 1, getRads = True,rtree = rtree[1], scan = scan)
                    retPointsr,retDistr,retRr = self.getVectorR(point, vec, n,tree[2],depth + 1, getRads = True,rtree = rtree[2], scan = scan)
                else:
                    retPointsl,retDistl = self.getVectorR(point, vec, n,tree[1],depth + 1, scan = scan)
                    retPointsr,retDistr = self.getVectorR(point, vec, n,tree[2],depth + 1, scan = scan)
                
                i = 0
                while i < len(retPointsl):
                    tretPoints.append(retPointsl[i])
                    tretDist.append(retDistl[i])
                    if getRads:
                        tretR.append(retRl[i])
                    i += 1
                i = 0
                while i < len(retPointsr):
                    tretPoints.append(retPointsr[i])
                    tretDist.append(retDistr[i])
                    if getRads:
                        tretR.append(retRr[i])
                    i += 1
                ntheta = getAngle(vec,nvec,getDistance(point,tp1),getDistance(point,node))
                #gets the node point if it falls inside the criteria
                if ntheta < (scan / 2): 
                    self.l += 1
                    tretPoints.append(node)
                    tretDist.append(getDistance(point,node))
                    if getRads:     
                        tretR.append(rnode)
                #Finally aquires the best n# of points from big list
                i = 0
                while i < len(tretPoints):
                    if i < n:
                        retPoints.append(tretPoints[i])
                        retDist.append(tretDist[i])
                        if getRads:
                            retR.append(tretR[i])
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
                            if getRads:
                                retR[tagj] = tretR[i]
                    i += 1
                
        if depth == 0:
            if getRads:
                return retPoints, retR
            else:
                return retPoints
        else:
            if getRads:
                return retPoints,retDist, retR
            else:
                return retPoints,retDist
    def treeLines(self,bounds : list,onode : list = [],side : int = 0,tree : list = [],depth : int = 0) -> list:
        #Works for 2D 
        #input: bounds format [[top left] , [bottom right]], both with [x,y]
        #output: [ [a1x,a1y] , [a2x,a2y] , [b1x,b1y] , [b2x,b2y] , ...] coupled points
        ret = []
        axis = depth % self.dimensions
        if depth == 0:
            tree = self.tree
            onode = [0,bounds[1][1]]
        node = tree[0]
        if axis == 0:#drawing straight along the y
            if len(tree) == 3 and node != 'Full':
                if side == 0:
                    ret.append([node[0],onode[1]])
                    ret.append([node[0],bounds[0][1]])
                else:
                    ret.append([node[0],onode[1]])
                    ret.append([node[0],bounds[1][1]])
                boundsleft = [bounds[0],[node[0],bounds[1][1]]]
                left = self.treeLines(boundsleft,node,1,tree[1],depth + 1)
                if left != None:
                    i = 0
                    while i < len(left):
                        ret.append([left[i][0],left[i][1]])
                        i = i + 1
                boundsright = [[node[0],bounds[0][1]],bounds[1]]
                right = self.treeLines(boundsright,node,0,tree[2],depth + 1)
                if right != None:
                    i = 0 
                    while i < len(right):
                        ret.append([right[i][0],right[i][1]])
                        i = i + 1
        else:#drwaing straight along the x
            
            if len(tree) == 3 and node != 'Full':
                if side == 0:
                    ret.append([onode[0],node[1]])
                    ret.append([bounds[1][0],node[1]])
                else:
                    ret.append([onode[0],node[1]])
                    ret.append([bounds[0][0],node[1]])
                boundsleft = [[bounds[0][0],node[1]],bounds[1]]
                left = self.treeLines(boundsleft,node,1,tree[1],depth + 1)
                i = 0
                if left != None:
                    while i < len(left):
                        ret.append([left[i][0],left[i][1]])
                        i = i + 1
                boundsright = [bounds[0],[bounds[1][0],node[0]]]
                right = self.treeLines(boundsright,node,0,tree[2],depth + 1)
                if right != None:
                    i = 0
                    while i < len(right):
                        ret.append([right[i][0],right[i][1]])
                        i = i + 1
        if ret != []:
            return ret



class SplitTree:
    #Split Tree is a versatile Quad/Oct tree designed for efficient stack storage for search
    def __init__(self,inpts,node:list,width: float,*,inrad : list = [],dim : int = 0):
        #Bounds are stored in a node(center [x,y,z]), heigh and width
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
        
    def addpoints(self, points : list = [],*,rads : float = 0.0):
        
        #First checks if the current leaf has been subdivided yet
        if not(self.state):
            if len(self.skelepts) + len(points) > self.maxpts:
                #If points more, than subdivide.
                self.state = True
                
                if isinstance(points,SkelePoint):
                    self.skelepts.append(points)
                else:
                    if rads == 0.0:
                        self.skelepts.append(SkelePoint(points))
                    else:
                        self.skelepts.append(SkelePoint(points,rad = rads))
                self.subdivide()
            else:
                if rads == 0.0:
                    if not(len(points) == 0):
                        self.skelepts.append(SkelePoint(points))
                else:
                    print(points,rads)
                    self.skelepts.append(SkelePoint(points,rad = rads))
                
        elif not(len(points) == 0):
            i = 0
            ne,se,nw,sw,a,b,c,d,e,f,g,h = [],[],[],[],[],[],[],[],[],[],[],[]
            ner,ser,nwr,swr,ar,br,cr,dr,er,fr,gr,hr = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
            if isinstance(points[0], float):
                points = [points]
            while i < len(points):
                if self.dim == 2:
                    if points[i][0] > self.node[0] and points[i][1] > self.node[1]:
                        ne = points[i]
                        if not(rads == 0.0):
                            ner = rads
                    elif points[i][0] > self.node[0] and points[i][1] < self.node[1]:
                        se = points[i]
                        if not(rads == 0.0):
                            ser = rads
                    elif points[i][0] < self.node[0] and points[i][1] > self.node[1]:
                        nw = points[i]
                        if not(rads == 0.0):
                            nwr = rads
                    else:
                        sw = points[i]
                        if not(rads == 0.0):
                            swr = rads
                else:
                    if points[i][0] > self.node[0] and points[i][1] > self.node[1] and points[i][2] > self.node[2]:
                        a = points[i]
                        if not(rads == 0.0):
                            ar = rads
                    elif points[i][0] > self.node[0] and points[i][1] > self.node[1] and points[i][2] < self.node[2]:
                        b = points[i]
                        if not(rads == 0.0):
                            br = rads
                    elif points[i][0] > self.node[0] and points[i][1] < self.node[1] and points[i][2] > self.node[2]:
                        c = points[i]
                        if not(rads == 0.0):
                            cr = rads
                    elif points[i][0] > self.node[0] and points[i][1] < self.node[1] and points[i][2] < self.node[2]:
                        d = points[i]
                        if not(rads == 0.0):
                            dr = rads
                    elif points[i][0] < self.node[0] and points[i][1] > self.node[1] and points[i][2] > self.node[2]:
                        e = points[i]
                        if not(rads == 0.0):
                            er = rads
                    elif points[i][0] < self.node[0] and points[i][1] > self.node[1] and points[i][2] < self.node[2]:
                        f = points[i]
                        if not(rads == 0.0):
                            fr = rads
                    elif points[i][0] < self.node[0] and points[i][1] < self.node[1] and points[i][2] > self.node[2]:
                        g = points[i]
                        if not(rads == 0.0):
                            gr = rads
                    else:
                        h = points[i]
                        if not(rads == 0.0):
                            hr = rads
                i += 1
            if self.dim == 2:
                if not(rads == 0.0):
                    self.leafs[0].addpoints(ne,rads=ner)
                    self.leafs[1].addpoints(se,rads=ser)
                    self.leafs[2].addpoints(nw,rads=nwr)
                    self.leafs[3].addpoints(sw,rads=swr)
                else:
                    self.leafs[0].addpoints(ne)
                    self.leafs[1].addpoints(se)
                    self.leafs[2].addpoints(nw)
                    self.leafs[3].addpoints(sw)
            else:
                if not(rads == 0.0):
                    self.leafs[0].addpoints(a,rads=ar)
                    self.leafs[1].addpoints(b,rads=br)
                    self.leafs[2].addpoints(c,rads=cr)
                    self.leafs[3].addpoints(d,rads=dr)
                    self.leafs[4].addpoints(e,rads=er)
                    self.leafs[5].addpoints(f,rads=fr)
                    self.leafs[6].addpoints(g,rads=gr)
                    self.leafs[7].addpoints(h,rads=hr)
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
                    return True, depth
                i += 1
            return False, depth
        return ret,dep
    
    def plot(self):
        if self.state == True:
            #If the figure is subdivided
            i = 0
            while i < len(self.leafs):
                self.leafs[i].plot()
                i += 1
            # plt.scatter(self.node[0],self.node[1],color='purple')
        else:
            # plt.plot([self.node[0] - self.width,self.node[0] - self.width],[self.node[1] + self.width,self.node[1] - self.width],5,color='orange')
            # plt.plot([self.node[0] + self.width,self.node[0] + self.width],[self.node[1] + self.width,self.node[1] - self.width],5,color='orange')
            # plt.plot([self.node[0] + self.width,self.node[0] - self.width],[self.node[1] + self.width,self.node[1] + self.width],5,color='orange')
            # plt.plot([self.node[0] + self.width,self.node[0] - self.width],[self.node[1] - self.width,self.node[1] - self.width],5,color='orange')
            if len(self.skelepts) > 0:
                i = 0
                tx = []
                ty = []
                while i < len(self.skelepts):
                    tx.append(self.skelepts[i].x)
                    ty.append(self.skelepts[i].y)
                    i += 1
                plt.scatter(tx,ty,5,color='green')
        
        

class SkelePoint:
#This is a class which has a point which holds x,y,z and r

    def __init__(self,point : list,*, rad : float,case : bool = False):
        self.x = point[0]
        self.y = point[1]
        self.r = rad
        if len(point) == 3:
            self.z = point[2]
            self.dimensions = 3
        else:
            self.dimensions = 2
        self.ordered = False
    
    def getPoint(self):
        if self.dimensions == 2:
            return [self.x,self.y]
        else:
            return [self.x,self.y,self.z]

    def getRad(self):
        return self.r

    def getCase(self):
        return self.case
    
    
    



















