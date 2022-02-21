# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:45 2022

@author: Graham Garcia
"""

import numpy as np
from DataStructures import normalize2D, kdTree, getDistance2D   
from random  import randint


def getRadius2D(point1, point2 , norm) -> float:
    distance = getDistance2D(point1, point2)
    mid = []
    for num1,num2 in zip(point1,point2):
        mid.append(np.abs(num1 - num2))
        #gives us an [x and a y] point mid
    top=[]    
    for num3,num4 in zip(mid,norm):
        top.append(num3*num4)
    Rtop = np.sqrt(np.power(top[0], 2) + np.power(top[1], 2))
    theta = np.arccos(Rtop / distance)#theta in radians
    radius = distance / (2*np.cos(theta))
    return radius
    
def Skeletize2D(points : list, norms : list):
    
    #skeletize takes in 
    #points, the given plain list of points [x,y] for 2D case
    #tree, a kd-tree means for quick searching for points
    #norms, a list of not yet normalized normal points [n_z,n_y] here for 2D case
    
    #then returns 2 things
    # finPoints = [[x,y],...] of skeleton points
    # finR = [r1,r2,...] of the radius of each skeleton point
    
    norms = normalize2D(norms)
    tree = kdTree(points, 2)
    
    finPoints = []#list of skeletized points
    finR = []#radius values of each point
    index = 1
    guessr = 0
    for point in points:
        print('Current Point:')
        print(point)
        #initialization
        tempr = []
        if index == 1:
            pcross = points[randint(index,len(points))]
            tempr.append(getRadius2D(point,pcross,norms[index - 1]))
        else:
            tempr.append(guessr)
        #print('guess radius')
        #print(tempr[0])
        #refinement
        i = 0
        centerp = []
        testp = []
        case = False
        while not case:
            
            #print(i)
            centerp = [float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])]
            #print(centerp)
            #print (index,point)
            testp = tree.getNear(centerp[0],centerp[1])
            tempr.append(getRadius2D(point, testp, norms[index - 1]))
            
            
            
            if tempr[len(tempr)-2] == tempr[len(tempr)-1]:
                finPoints.append(centerp)
                finR.append(tempr[i+1])
                #print('found')
                case = True
            
            
            i = i + 1
        #guess values
        if index != len(points):
            pointq = points[index]
            guessr = getRadius2D(pointq, testp, norms[index])
            
        print(index-1)
        print('/')
        print(len(points))
        print((index-1) / len(points))    
        index = index + 1
        
        
    #returns important values    
    return finPoints,finR

