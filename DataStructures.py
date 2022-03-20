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
    
    
    
    
    
    #non recursive function 
    def getNear(self,pointx,pointy,RefPoint):
        point = [pointx,pointy]
        retPoint, dep, distance = self.getNearR(point,RefPoint,self.tree)
        return retPoint    
        
               
    def getNearR(self,searchPoint : list , exclude : list , tree : list, depth : int = 0):
        #norm will help us find the closeest point twards the center 
        
        point = []
        tdep = depth
        Tdist = 10 * pow(10,10)
        # print('go')
        if tree[0] == 'Full':#if it is bottom layer return 
            
            if len(tree) > 2:
                i = 1
                while i < len(tree):
                    if (tree[i][0] <= searchPoint[0] and tree[i][0] >= exclude[0]) or (tree[i][0] >= searchPoint[0] and tree[i][0] <= exclude[0]):
                        if (tree[i][1] <= searchPoint[1] and tree[i][1] >= exclude[1]) or (tree[i][1] >= searchPoint[1] and tree[i][1] <= exclude[1]):
                            td = getDistance2D(searchPoint,tree[i])
                            if td < Tdist and tree[i] != exclude:
                                point = tree[i]
                                Tdist = td
                    i = i + 1
                if point == []:
                    point = 'None'
            elif len(tree) == 2 and tree[1] != exclude:
                point = tree[1]
                Tdist = 0
            else:
                point = 'None'
                tdep = 0
                Tdist = 0
                
        elif tree[0] == searchPoint or tree[0] == exclude:#if search is a point on list
            point1,tdep1,tdist1 = self.getNearR(searchPoint,exclude,tree[1],depth + 1)
            point2,tdep2,tdist2 = self.getNearR(searchPoint,exclude,tree[2],depth + 1)
            if tdist1 > tdist2:
                Tdist = tdist2
                point = point2
                tdep = tdep2
            else:
                Tdist = tdist1
                point = point1
                tdep = tdep1
                
                
        else:#needs to search further
            # print('depth',depth)
            # print('searchpoint',searchPoint)
            # print('exclude',exclude)
            #first will grab lowest point from underneath
            axis = depth % self.dimensions
            if searchPoint[axis] >= tree[0][axis]:#will prioritize 'right' side of tree
                # print('left',tree[2])
                # print('axis',axis)
                # print()
                point,tdep,Tdist = self.getNearR(searchPoint,exclude,tree[2],depth + 1)
            else:#if 'left' side of tree
                # print('right',tree[1])
                # print('axis',axis)
                # print()
                point,tdep,Tdist = self.getNearR(searchPoint,exclude,tree[1],depth + 1)
            
            # print('checking')
            if point == 'None' or (point[0] <= searchPoint[0] and point[0] >= exclude[0]) or (point[0] >= searchPoint[0] and point[0] <= exclude[0]):
                # print('xlim',point)
                if (point == 'None' or point[1] <= searchPoint[1] and point[1] >= exclude[1]) or (point[1] >= searchPoint[1] and point[1] <= exclude[1]):
                    # print('ylim')
                    if tdep - depth < self.dimensions + 3:
                        # print('search',searchPoint, 'exclude', exclude)
                        Ndist = getDistance2D(tree[0], searchPoint)
                        if Ndist < Tdist:
                            point = tree[0]
                            tdep = depth
                            Tdist = Ndist
        # print('found point',point)
        return point, tdep , Tdist
                
        def treeLines2D(self,bounds : list,onode : list = [],side : int = 0,depth : int = 0) -> list:
            #bounds format [[top left] , [bottom right]], both with [x,y]
            #output: [ [a1x,a1y] , [a2x,a2y] , [b1x,b1y] , [b2x,b2y] , ...] coupled points
            ret = []
            axis = depth % self.dimensions
            if depth == 0:
                onode = []
            node = self.tree[0]
            if self.tree[0] != 'Full':
                if axis == 0:#drawing straight along the y
                    if side == 0 #right side
                        ret.append([node[0],bounds[0][1]])
                        ret.append([node[0],onode[1]])
                    else: #leftside
                        ret.append([node[0],onode[1]])
                        ret.append([node[0],bounds[1][1]])
                else:#drwaing straight along the x
                    if side == 0:
                        ret.append([bounds[0][0],node[1]])
                        ret.append([onode[0],node[1]])
                    else:
                        ret.append([onode[0],node[1]])
                        ret.append([bounds[1][0],node[1]])
                    temp = self.treeLines2D(bounds,node,)
                
            
        
        