# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:45 2022

@author: Graham Garcia
"""

import numpy as np
from DataStructures import normalize2D, normalize3D, kdTree, getDistance2D, getDistance3D
from random  import randint
import matplotlib
import matplotlib.pyplot as plt

def getRadius2D(point1, point2 , norm) -> float:
    
    
    distance = getDistance2D(point1, point2)
    Pvec = [point1[0] - point2[0] , point1[1] - point2[1]]
    top = norm[0] * Pvec[0] + norm[1] * Pvec[1]
    if top/distance <= 1 and top/distance >= -1:
        top = top/distance
        theta = np.arccos(top)
    else:
        print('rip',top/distance)
        top = top/distance
        top = top + 1
        top = top % 2
        top = top - 1  
        theta = np.arccos(top)
    
    radius = np.abs(distance / (2 * np.cos(theta)))
    return radius

def getRadius3D(point1,point2,norm) -> float:
    return 0
    
def Skeletize2D(points : list, norms : list):
    #skeletize takes in 
    #points, the given plain list of points [x,y] for 2D case
    #norms, a list of not yet normalized normal points [n_z,n_y] here for 2D case
    
    #then returns 2 things
    # finPoints = [[x,y],...] of skeleton points
    # finR = [r1,r2,...] of the radius of each skeleton point
    
   
    
    pts = []
    i = 0
    while i < len(points):
        pts.append(points[i])
        i = i + 1
        
    norms = normalize2D(norms)
    tree = kdTree(pts, 2)
    
    #setting the distance threshhold
    close = tree.getNearR(points[0],points[0])
    threshDistance = getDistance2D(points[0],close)
    
    finPoints = []#list of skeletized points
    finR = []#radius values of each point
    index = 1
    guessr = 0
    for point in points:
        tempr = []
        if index == 1:
            pcross = points[randint(index,len(points))]
            tempr.append(getRadius2D(point,pcross,norms[index - 1]))
        else:
            # print('guessr',guessr)
            tempr.append(guessr)
        
        i = 0
        centerp = []
        testp = []
        case = False
        
        #solve loop
        while not case:
            print(index - 1, i,tempr[len(tempr)-1])
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
            testp = tree.getNearR(centerp[len(centerp)-1], point)
            tempr.append(np.round(getRadius2D(point, testp, norms[index - 1]),6))
            leng = len(tempr)-1
            
            if tempr[leng] < 2 * threshDistance:
                case = True
            
            #cases for cacthing when stuck  
            if tempr[leng] == tempr[leng - 1] and i > 1:
                centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
                finPoints.append(centerp[len(centerp)-1])
                finR.append(tempr[leng])
                case = True
            #going back and fourth from two radii    
            if i >= 3:
                if tempr[i] == tempr[i-2] and tempr[i-1] == tempr[i-3]:
                    if tempr[i] > tempr[i-1]:
                        finPoints.append(centerp[i-1])
                        finR.append(tempr[i-1])
                        case = True
                    else:
                        finPoints.append(centerp[i])
                        finR.append(tempr[i])
                        case = True
                
            i = i + 1
            
        #guess values
        if index  != len(points):
            guessr = finR[len(finR)-1] * 100        
        index = index + 1
        
    
        
    #returns important values 
        
        
    return finPoints,finR

def Skeletize3D(points : list, norms : list):
    #skeletize takes in 
    #points, the given plain list of points [x,y,z] for 3D case
    #norms, a list of not yet normalized normal points [n_z,n_y,n_z] here for 3D case
    
    #then returns 2 things
    # finPoints = [[x,y,z],...] of skeleton points
    # finR = [r1,r2,...] of the radius of each skeleton point
    
   
    
    pts = []
    i = 0
    while i < len(points):
        pts.append(points[i])
        i = i + 1
        
    norms = normalize3D(norms)
    tree = kdTree(pts, 3)
    
    #setting the distance threshhold
    close = tree.getNearR(points[0],points[0])
    threshDistance = getDistance3D(points[0],close)
    
    finPoints = []#list of skeletized points
    finR = []#radius values of each point
    index = 1
    guessr = 0
    for point in points:
        tempr = []
        if index == 1:
            pcross = points[randint(index,len(points))]
            tempr.append(getRadius3D(point,pcross,norms[index - 1]))
        else:
            # print('guessr',guessr)
            tempr.append(guessr)
        
        i = 0
        centerp = []
        testp = []
        case = False
        
        #solve loop
        while not case:
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1]),float(point[2]-norms[index-1][2]*tempr[len(tempr)-1])])
            testp = tree.getNearR(centerp[len(centerp)-1], point)
            tempr.append(np.round(getRadius3D(point, testp, norms[index - 1]),6))
            leng = len(tempr)-1
            
            if tempr[leng] < 2 * threshDistance:
                case = True
            
            #cases for cacthing when stuck  
            if tempr[leng] == tempr[leng - 1] and i > 1:
                centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1]),float(point[2]-norms[index-1][2]*tempr[len(tempr)-1])])
                finPoints.append(centerp[len(centerp)-1])
                finR.append(tempr[leng])
                case = True
            #going back and fourth from two radii    
            if i >= 3:
                if tempr[i] == tempr[i-2] and tempr[i-1] == tempr[i-3]:
                    if tempr[i] > tempr[i-1]:
                        finPoints.append(centerp[i-1])
                        finR.append(tempr[i-1])
                        case = True
                    else:
                        finPoints.append(centerp[i])
                        finR.append(tempr[i])
                        case = True
                
            i = i + 1
            
        #guess values
        if index  != len(points):
            guessr = finR[len(finR)-1] * 100        
        index = index + 1
        
    
        
    #returns important values 
        
        
    return finPoints,finR

def BuildSkeleton(centerPoints : list, radius : list, dim):
    
    #first define the geometric center of all the points, averaging out, then the closeest point will be the node
    if dim == 2:#2d
        i = 0
        tx = 0
        ty = 0
        n = len(centerPoints)
        while i < n:
            tx = tx + centerPoints[i][0]
            ty = ty + centerPoints[i][1]
            i = i + 1
        tgeoCent = [tx/n,ty/n]
        tree = kdTree(centerPoints,2)
        geoCent = tree.getNearR(tgeoCent,[10000,10000])
    else:#3D 
        i = 0
        tx = 0
        ty = 0
        tz = 0
        n = len(centerPoints)
        while i < n:
            tx = tx + centerPoints[i][0]
            ty = ty + centerPoints[i][1]
            tz = tz + centerPoints[i][2]
            i = i + 1
        tgeoCent = [tx/n,ty/n,tz/n]
        tree = kdTree(centerPoints,3)
        geoCent = tree.getNearR(tgeoCent,[10000,10000])
    #now a final point will be classified as the center point and we can branch out 
    