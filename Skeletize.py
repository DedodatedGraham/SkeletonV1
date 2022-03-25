# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:45 2022

@author: Graham Garcia
"""

import numpy as np
from DataStructures import normalize2D, kdTree, getDistance2D   
from random  import randint


def getRadius2D(point1, point2 , norm) -> float:
    
    # print('P',point1)
    # print('P`',point2)
    # print('Norm',norm)
    distance = getDistance2D(point1, point2)
    # print('Distance from P to P`',distance)
    Pvec = [point1[0] - point2[0] , point1[1] - point2[1]]
    # print('Vector from P to P`',Pvec)
    top = norm[0] * Pvec[0] + norm[1] * Pvec[1]
    # print('Dot Product',top)
    if top/distance <= 1 and top/distance >= -1:
        top = top/distance
        # print('inside',top)
        theta = np.arccos(top)
        # print('angle',theta)
    else:
        top = top/distance
        # print('inside1',top)
        top = top + 1
        # print('inside2',top)
        top = top % 2
        # print('inside3',top)
        top = top - 1
        # print('inside4',top)    
        theta = np.arccos(top)
        # print('angle',theta)
    
    radius = np.abs(distance / (2 * np.cos(theta)))
    #print('radius',radius)
    #print()
    return radius
    
def Skeletize2D(points : list, norms : list):
    #skeletize takes in 
    #points, the given plain list of points [x,y] for 2D case
    #tree, a kd-tree means for quick searching for points
    #norms, a list of not yet normalized normal points [n_z,n_y] here for 2D case
    
    #then returns 2 things
    # finPoints = [[x,y],...] of skeleton points
    # finR = [r1,r2,...] of the radius of each skeleton point
    counts = []
    pts = []
    i = 0
    while i < len(points):
        pts.append(points[i])
        i = i + 1
        
    norms = normalize2D(norms)
    tree = kdTree(pts, 2)
    finPoints = []#list of skeletized points
    finR = []#radius values of each point
    index = 1
    guessr = 0
    for point in points:
        tempr = []
        if index == 1:
            pcross = points[randint(index,len(points))]
            pcross = points[100]
            tempr.append(getRadius2D(point,pcross,norms[index - 1]))
        else:
            tempr.append(guessr)
        
        
        
            
        i = 0
        centerp = []
        testp = []
        case = False
        while not case:
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
            testp = tree.getNearR(centerp[len(centerp)-1], point)
            tempr.append(getRadius2D(point, testp, norms[index - 1]))
            
            leng = len(tempr)-1
            if tempr[leng] == tempr[leng - 1]:
                finPoints.append(centerp[len(centerp)-1])
                finR.append(tempr[leng])
                case = True
            
                
            i = i + 1
        #guess values
        
        if index  != len(points):
            pointq = points[index]
            case = True
            i = 1
            while case:
                if pointq == testp:
                    testp = points[index + i]
                    i = i + 1
                else:
                    case = False
            guessr = getRadius2D(pointq,testp, norms[index])
        
        index = index + 1
        
        
        
    #returns important values 
        
        
    return finPoints,finR

