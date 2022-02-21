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
    return np.sqrt(np.abs(pow(point1[0]-point2[0],2)) + np.abs(pow(point1[1]-point2[1],2)))
    
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
        #print(point)
        retPoint, dep, distance = self.getNearR(point,self.tree)
        #print(dep)
        #print(distance)
        return retPoint    
        
               
    def getNearR(self,searchPoint : list, tree : list, depth : int = 0):
        point = []
        
        if tree[0] == 'Full':#if bottom layer which is marked must brute force and compare
            #print('Full')
            index = 1
            i = 1
            #print(getDistance2D(searchPoint, tree[i]))
            if getDistance2D(searchPoint, tree[i]) == 0:
                i = i + 1
            if self.dimensions == 2 :
                Trad = getDistance2D(searchPoint,tree[i])
                while i < len(tree):
                    if getDistance2D(searchPoint,tree[i]):
                        Trad = getDistance2D(searchPoint, tree[i]);
                        index = i
                        i = i + 1
            else:
                Trad = getDistance3D(searchPoint,tree[i])
                while i < len(tree):
                    if getDistance3D(searchPoint,tree[i]) < Trad:
                        Trad = getDistance2D(searchPoint, tree[i]);
                        index = i
                        i = i + 1
            return tree[index], depth , Trad       
                
        
        elif :
        
            
        else:#needs to go deeper
            axis = (depth % self.dimensions)
            if searchPoint[axis] < tree[0][axis] : #if is "left" of node
                #print(depth)
                #print('left')
                #print(tree)
                point, td, Trad = self.getNearR(searchPoint, tree[1], depth + 1)
                if Trad > getDistance2D(searchPoint,tree[0]) and td - depth < 3:
                    #print('replace')
                    point = tree[0]
                    td = depth
                    Trad = getDistance2D(searchPoint, tree[0])
                #print(depth)
                #print('left-up')
            else:
                #print(depth)
                #print('right')
                #print(tree)
                point, td, Trad = self.getNearR(searchPoint,tree[2],depth + 1)  
                if Trad > getDistance2D(searchPoint,tree[0]) and td - depth < 3:
                    #print('replace')
                    point = tree[0]
                    td = depth
                    Trad = getDistance2D(searchPoint, tree[0])
                #print(depth)
                #print('right-up')
        return point, td , Trad
                
        
        