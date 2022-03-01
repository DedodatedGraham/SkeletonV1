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
        
        
        
    def makeTree(self,points:list, depth : int) -> list:
        finTree = []
        #designed to never have empty leafs
        if self.dimensions == 2:#formatting for 2D
            if len(points) > 5 and depth < 100:
                axis = depth % self.dimensions #gets axis to divide along
                points = np.sort(points,axis)
            
                mid = len(points) // 2
                
                finTree.append([points[mid][0],points[mid][1]])#choses node point
                finTree.append(self.makeTree(points[:mid],depth+1))#gets roots(Left of node)
                finTree.append(self.makeTree(points[mid+1:],depth+1))#(right of node)
                
            
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
    def getNear(self,pointx,pointy):
        point = [pointx,pointy]
        retPoint, dep, distance = self.getNearR(point,self.tree)
        
        return retPoint    
        
               
    def getNearR(self,searchPoint : list, tree : list, depth : int = 0):
        point = []
        tdep = depth
        Tdist = 10 * pow(10,10)
        if tree[0] == 'Full':#if it is bottom layer return 
            
            if len(tree) > 2:
                i = 1
                while i < len(tree):
                    td = getDistance2D(searchPoint,tree[i])
                    if(td < Tdist):
                        point = tree[i]
                        Tdist = td
                    i = i + 1
            elif len(tree) == 2:
                point = tree[1]
                Tdist = 0
            else:
                point = 'None'
                tdep = 0
                Tdist = 0
                
        elif tree[0] == searchPoint:#if search is a point on list
            point1,tdep1,tdist1 = self.getNearR(searchPoint,tree[1],depth + 1)
            point2,tdep2,tdist2 = self.getNearR(searchPoint,tree[0],depth + 1)
            if tdist1 > tdist2:
                Tdist = tdist2
                point = point2
                tdep = tdep2
            else:
                Tdist = tdist1
                point = point1
                tdep = tdep1
        else:#needs to search further
            #first will grab lowest point from underneath
            axis = depth % self.dimensions
            if searchPoint[axis] >= tree[0][axis]:#will prioritize 'right' side of tree
                
                point,tdep,Tdist = self.getNearR(searchPoint,tree[2],depth + 1)
            else:#if 'left' side of tree
                
                point,tdep,Tdist = self.getNearR(searchPoint,tree[1],depth + 1)
            
            #then if within a count of  one dimensions away
            if tdep - depth < self.dimensions + 3:
                
                Ndist = getDistance2D(tree[0], searchPoint)
                if Ndist < Tdist:
                    point = tree[0]
                    tdep = depth
                    Tdist = Ndist
        
        return point, tdep , Tdist
                
        
        