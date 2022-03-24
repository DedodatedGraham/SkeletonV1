# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:46:08 2022

@author: graha
"""


import numpy as np

def normalize2D(points : list) -> list:
    #each point is [x,y]
    i = 0
    retpoints = []
    while i < len(points):
        
        tempx = points[i][0]
        tempy = points[i][1]
        normalize = 1/np.sqrt(np.power(tempx,2)+np.power(tempy,2))
        tempx = tempx * normalize
        tempy = tempy * normalize
        retpoints.append([tempx,tempy])
        i = i + 1
    return retpoints

def getDistance2D(point1, point2) -> float:
    return np.sqrt(pow(np.abs(point1[0]-point2[0]),2) + pow(np.abs(point1[1]-point2[1]),2))
    
def getDistance3D(point1, point2) -> float:
    return np.sqrt(np.abs(pow(point1[0]-point2[0],2)) + np.abs(pow(point1[1]-point2[1],2))+np.abs(pow(point1[2]-point2[2],2)))
class kdTree:
    
    def __init__(self,points : list, dim : int):
        
    #for sorting points along an axis
        self.dimensions = dim
        self.tree = self.makeTree(points,0)
        
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
                #print('level points',points)
                finTree.append([points[mid][0],points[mid][1]])#choses node point
                # print('Midpoint',finTree[0])
                pointsl = []
                i = 0
                while i < mid:
                    pointsl.append([points[i][0] , points[i][1]])
                    i = i + 1
                #print('Left points',pointsl)
                finTree.append(self.makeTree(pointsl,depth+1))#gets roots(Left of node)
                
                pointsr = []
                i = mid + 1
                
                while i < len(points):
                    pointsr.append([points[i][0] , points[i][1]])
                    i = i + 1
                #print('Right points',pointsr)
                finTree.append(self.makeTree(pointsr,depth+1))#(right of node)
                
                
            
            else:
                #add in all points, if small amount of points or too deep
                i = 0
                finTree.append('Full')#marks a bottom layer
                while i < len(points):
                    finTree.append([points[i][0],points[i][1]])
                    i = i + 1
        if self.dimensions == 3:#formatting for 3D
            if len(points) > 5 and depth < 100:
                axis = depth % self.dimensions #gets axis to divide along
                points = np.sort(points,axis)
            
                mid = len(points) // 2
                
                finTree.append('Full')
                finTree.append([points[mid][0],points[mid][1]],points[mid][2])#choses node point
                finTree.append(self.makeTree(points[:mid],depth+1))#gets roots
                finTree.append(self.makeTree(points[mid+1:],depth+1))
            
            
            else:
                #add in all points, if small amount of points or too deep
                i = 0
                while i < len(points):
                    finTree.append([points[i][0],points[i][1],points[i][2]])
                    i = i + 1
        return finTree
    
      
    #searches tree              
    def getNearR2D(self,searchPoint : list , exclude : list ,tree : list = [], depth : int = 0):
        dmin = 0
        tdep = 0
        smallestLayer = []
        if depth == 0:
            tree= self.tree
        node = tree[0]
        axis = depth % self.dimensions
        if node == 'Full':
            tdep = depth
            #if searching on a bottom Layer, will brute force
            i = 1
            while i < len(tree):
                if(tree[i] != exclude):
                    if i == 1:
                        dmin = getDistance2D(searchPoint, tree[1])
                        smallestLayer = tree[1]
                    dtemp = getDistance2D(searchPoint,tree[i])
                    if dtemp < dmin:
                        dmin = dtemp
                        smallestLayer = tree[i]
                i = i + 1
        else:
            if tree[0] == exclude:
                if searchPoint[axis] <= tree[0][axis]:
                    #want to go deeper to the left
                    pLeft,tdep,dmin = self.getNearR2D(searchPoint,exclude,tree[1], depth + 1)
                else:
                    #want to go deeper to the right
                    pRight,tdep,dmin = self.getNearR2D(searchPoint,exclude,tree[2], depth + 1)
            else:
                if searchPoint[axis] <= tree[0][axis]:
                    #want to go deeper to the left
                    pLeft,tdep,dmin = self.getNearR2D(searchPoint,exclude,tree[1], depth + 1)
                else:
                    #want to go deeper to the right
                    pRight,tdep,dmin = self.getNearR2D(searchPoint,exclude,tree[2], depth + 1)
                #makes sure node isnt closer, only needs to be ran for non excluded points
                if tdep - depth < 3:
                    tdis = getDistance2D(searchPoint,node)
                    if tdis < dmin:
                        smallestLayer = node
                        tdep = depth
                        dmin = tdis
                if axis == 0:
                    tpoint = []
                else:
                    tpoint = []
                tdis = getDistance2D(searchPoint,tpoint)
            
                
                        
        return smallestLayer, tdep, dmin
                    
                
        
        
        
        
                
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
        
        