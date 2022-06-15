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

from Skeletize import getDistance

class kdTree:
    
    def __init__(self,points : list, dim : int):
        
    #for sorting points along an axis
        self.PointsList = []
        i = 0
        while i < len(points):
            self.PointsList.append(points[i])
            i = i + 1
        self.dimensions = dim
        self.tree = self.makeTree(self.PointsList,0)
        
    def sort(self,points : list, dimension) -> list:
        
       
        for i in range(len(points)):
            min_i = i
            for j in range(i+1,len(points)):
                if points[j][dimension] < points[min_i][dimension]:
                    min_i = j
            points[i],points[min_i] = points[min_i],points[i]
        
        return points
    
    #contructs the tree    
    def makeTree(self,points:list, depth : int) -> list:
        finTree = []
        #designed to never have empty leafs
        if self.dimensions == 2:#formatting for 2D
            if len(points) > 5 and depth < 100:
                axis = depth % self.dimensions #gets axis to divide along
                points = self.sort(points,axis)
                
                
                mid = len(points) // 2
                finTree.append([points[mid][0],points[mid][1]])#choses node point
                pointsl = []
                i = 0
                while i < mid:
                    pointsl.append([points[i][0] , points[i][1]])
                    i = i + 1
                finTree.append(self.makeTree(pointsl,depth+1))#gets roots(Left of node)
                
                pointsr = []
                i = mid + 1
                
                while i < len(points):
                    pointsr.append([points[i][0] , points[i][1]])
                    i = i + 1
                finTree.append(self.makeTree(pointsr,depth+1))#(right of node)
                
            else:
                #add in all points, if small amount of points or too deep
                i = 0
                finTree.append('Full')#marks a bottom layer
                while i < len(points):
                    finTree.append([points[i][0],points[i][1]])
                    i = i + 1
        else:
            if len(points) > 5 and depth < 100:
                axis = depth % self.dimensions #gets axis to divide along
                points = self.sort(points,axis)
                
                
                mid = len(points) // 2
                finTree.append([points[mid][0],points[mid][1],points[mid][2]])#choses node point
                pointsl = []
                i = 0
                while i < mid:
                    pointsl.append([points[i][0] , points[i][1] , points[i][2]])
                    i = i + 1
                finTree.append(self.makeTree(pointsl,depth+1))#gets roots(Left of node)
                
                pointsr = []
                i = mid + 1
                
                while i < len(points):
                    pointsr.append([points[i][0] , points[i][1] , points[i][2]])
                    i = i + 1
                finTree.append(self.makeTree(pointsr,depth+1))#(right of node)
                
            else:
                #add in all points, if small amount of points or too deep
                i = 0
                finTree.append('Full')#marks a bottom layer
                while i < len(points):
                    finTree.append([points[i][0],points[i][1],points[i][2]])
                    i = i + 1
            
        return finTree
    
      
    #searches tree              
    def getNearR(self,searchPoint : list , exclude : list ,tree : list = [], depth : int = 0):
        smallestLayer = []
        dmin = 0
        if depth == 0:
            tree= self.tree
        node = tree[0]
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
                    dtemp = getDistance(searchPoint,tree[i])
                    if dtemp < dmin and dtemp != 0:
                        dmin = dtemp
                        smallestLayer = tree[i]
                i = i + 1
        else:
            
            if searchPoint[axis] <= tree[0][axis]:
                #want to go deeper to the left
                smallestLayer = self.getNearR(searchPoint,exclude,tree[1], depth + 1)
            else:
                #want to go deeper to the right
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
                    tdis = tdis1
                
            #checks if the circle crosses the plane and needs to check other leafs on the other side
            #reduces some of the math/ steps needed because i made it find the axis distance from the search 
            #point to the node. so if theres a possibility there could be a point itll check it but have to do less math
            #and back and fourth sending to make it happen
            tPoint = None
            if searchPoint[axis] <= tree[0][axis]:
                tdis3 = node[axis] - searchPoint[axis]
                if tdis3 <= tdis and tdis3 != 0:
                    tPoint = self.getNearR(searchPoint,exclude,tree[2], depth + 1)
            else:
                tdis3 = searchPoint[axis] - node[axis]
                if tdis3 <= tdis and tdis3 != 0:
                    tPoint = self.getNearR(searchPoint,exclude,tree[1], depth + 1)
            if tPoint != None and tPoint != []:
                tdis2 = getDistance(searchPoint,tPoint)
                if tdis2 < tdis and tdis2 != 0:
                    smallestLayer = tPoint
                    tdis = tdis2
        return smallestLayer
                    
                
        
        
        
        
                
    def treeLines2D(self,bounds : list,onode : list = [],side : int = 0,tree : list = [],depth : int = 0) -> list:
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
                left = self.treeLines2D(boundsleft,node,1,tree[1],depth + 1)
                if left != None:
                    i = 0
                    while i < len(left):
                        ret.append([left[i][0],left[i][1]])
                        i = i + 1
                boundsright = [[node[0],bounds[0][1]],bounds[1]]
                right = self.treeLines2D(boundsright,node,0,tree[2],depth + 1)
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
                left = self.treeLines2D(boundsleft,node,1,tree[1],depth + 1)
                i = 0
                if left != None:
                    while i < len(left):
                        ret.append([left[i][0],left[i][1]])
                        i = i + 1
                boundsright = [bounds[0],[bounds[1][0],node[0]]]
                right = self.treeLines2D(boundsright,node,0,tree[2],depth + 1)
                if right != None:
                    i = 0
                    while i < len(right):
                        ret.append([right[i][0],right[i][1]])
                        i = i + 1
        if ret != []:
            return ret
