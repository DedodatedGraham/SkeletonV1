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
    diffx= np.abs(point1[0]-point2[0])
    diffy= np.abs(point1[1]-point2[1])
    px = 0
    py = 0
    if point1[0] > point2[0]:
        px = point2[0] + diffx
    else:
        px = point1[0] + diffx
    if point1[1] > point2[1]:
        py = point2[1] + diffy
    else:
        py = point1[1] + diffy
    inside = np.sqrt(pow(px * norm[0],2) + pow(py * norm[1],2))
    if inside != 0:
        inside = inside/distance 
    if inside < 1 and inside > -1:
        theta = np.arccos(inside)
        radius = distance/(2*np.cos(theta))
        
    elif inside > 0:
        factor = inside % 1
        theta = np.arccos(factor)
        radius = distance/(2*np.cos(theta))
    else:
        factor = (-1 * inside) % 1
        theta = np.arccos(factor)
        radius = distance/(2*np.cos(theta))
        
    
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
        #print('Current Point:')
        #print(point)
        #initialization
        
        
        tempr = []
        if index == 1:
            pcross = points[randint(index,len(points))]
            tempr.append(getRadius2D(point,pcross,norms[index - 1]))
        else:
            tempr.append(guessr)
        
        
        print(index-1)
        i = 0
        centerp = []
        testp = []
        case = False
        while not case:
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
            #print ('itteration, point: ',i, ' , ', point)
            
            testp = tree.getNear(centerp[i][0],centerp[i][1])
            
                        
                        
                        
                        
            
            tempr.append(getRadius2D(point, testp, norms[index - 1]))
            
                
            
            if tempr[len(tempr)-1] == tempr[len(tempr) - 2]:
                finPoints.append(centerp[len(centerp)-1])
                finR.append(tempr[i+1])
                #print('found')
                case = True
            
            
                
            
                
            i = i + 1
        #guess values
        if index != len(points):
            pointq = points[randint(0,index)]
            
            if(pointq[0] == points[index][0] and pointq[1] == points[index][1]):
               pointq = points[randint(0,index - 1)]
            guessr = getRadius2D(pointq, points[index], norms[index])
            
        index = index + 1
        
        
    #returns important values    
    return finPoints,finR

