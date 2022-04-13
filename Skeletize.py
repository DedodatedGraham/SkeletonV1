# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:45 2022

@author: Graham Garcia
"""

import numpy as np
from DataStructures import normalize2D, kdTree, getDistance2D   
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
        top = top/distance
        top = top + 1
        top = top % 2
        top = top - 1  
        theta = np.arccos(top)
    
    radius = np.abs(distance / (2 * np.cos(theta)))
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
            tempr.append(getRadius2D(point,pcross,norms[index - 1]))
        else:
            # print('guessr',guessr)
            tempr.append(guessr)
        
        
        
        treesx = []
        treesy = []
        i = 0
        centerp = []
        testp = []
        case = False
        while not case:
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
            testp = tree.getNearR(centerp[len(centerp)-1], point)
            tempr.append(np.round(getRadius2D(point, testp, norms[index - 1]),6))
            leng = len(tempr)-1
            # print(index,i)
            # if index == 17:
            #     plt.plot(testp[0],testp[1])
            #     print(index-1 , i)
            #     print('Point',point,'norm',norms[index-1])
            #     print('center points',centerp[len(centerp)-1])
            #     print('tree point',testp)
            #     print('radius',tempr[len(tempr)-1])
            #     print()
                # treesx.append(testp[0])
                # treesy.append(testp[1])
                # if i > 0:
                #     theta = np.linspace(0, 2*np.pi, 100)
                #     r = tempr[leng]
                #     x1 = centerp[len(centerp)-1][0] + r*np.cos(theta)
                #     x2 = centerp[len(centerp)-1][1] + r*np.sin(theta)
                #     plt.plot(x1, x2)
                
                
            if tempr[leng] == tempr[leng - 1] and i > 1:
                centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
                finPoints.append(centerp[len(centerp)-1])
                finR.append(tempr[leng])
                if index == 420:
                    cx = []
                    cy = []
                    j = 1
                    while j < len(centerp):
                        cx.append(centerp[j][0])
                        cy.append(centerp[j][1])
                        j = j + 1
                    plt.scatter(cx,cy)
                    
                    plt.scatter(treesx,treesy)
                case = True
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

