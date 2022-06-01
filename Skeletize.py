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
import scipy
import pandas as pd

def checkRepeat(check : list):
    n = 2 #order of repeat
    a = False
    size = len(check)
    while n < size:
        
        right = check[size - n : size]
        left = check[size - 2 * n : size - (n)]
        
        if right == left:
            a = True
            break
        n = n + 1
    return a , n


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

def getRadius3D(point1,point2,norm) -> float:
    return 0


def thin2D(opts : list, measured : list, finPts : list, finR : list, pointDis):
    pts = []
    j = 0
    #interface points
    while j < len(opts):
        pts.append(opts[j])
        j = j + 1
    
    # N = 5 #points to grab arround test point
    Threshold = pointDis  # applies for how much varying is allowed between a real point / fake point
    
    
    thin1p = []
    thin1r = []
    #removing the repeated points, as they arent very important 
    i = 0
    while i < len(finPts) - 1:
        # print(i,'/',len(finPts))
        j = i + 1
        noRepeat = True
        while j < len(finPts):
            if np.abs(finR[i] - finR[j]) <= Threshold/100:
                if np.abs(finPts[i][0] - finPts[j][0]) <= Threshold/100:
                    if np.abs(finPts[i][1] - finPts[j][1]) <= Threshold/100:
                        noRepeat = False
                        
            j = j + 1
            
            
        if noRepeat:
            thin1p.append(finPts[i])
            thin1r.append(finR[i])
        i = i + 1
    print(len(thin1p))
    return thin1p,thin1r
    
    
def Skeletize2D(points : list, norms : list):
    #skeletize takes in 
    #points, the given plain list of points [x,y] for 2D case
    #norms, a list of not yet normalized normal points [n_x,n_y] here for 2D case
    
    #then returns 2 things
    # finPoints = [[x,y],...] of skeleton points
    # finR = [r1,r2,...] of the radius of each skeleton point
    
    level = 96
    
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
    
    #TOLLERANCES
    cweight = 0.5#weight of c value(exponetialy weighted)
    maxReach = 11 #max ittrations of the reach algorithm
    pointtol = 0.2 #in %
    closew = 100 #Another modifier for tollerance on skeleton filter
    
    thinPoints = []
    finPoints = []#list of skeletized points
    finR = []#radius values of each point
    index = 1
    guessr = 0
    for point in points:
        
        tempr = []
        if index == 1:
            pcross = points[randint(index,len(points))]
            tempr.append(np.round(getRadius2D(point,pcross,norms[index - 1]),6))
        else:
            # print('guessr',guessr)
            tempr.append(guessr)
        
        i = 0
        centerp = []
        testp = []
        case = False
        print(index, '/' , len(points))
        #solve loop
        while not case:
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
            testp = tree.getNearR(centerp[len(centerp)-1], point)
            tempr.append(np.round(getRadius2D(point, testp, norms[index - 1]),6))
            leng = len(tempr)-1
            
            if tempr[leng] < 2 * threshDistance:
                case = True
            
            #cases for cacthing when stuck  
            if tempr[leng] == tempr[leng - 1] and i > 1:
                
                reach = True
                reachitt = 1
                while reach:
                    c = np.power(10, reachitt) * threshDistance * np.power(cweight,reachitt)
                    if(reachitt > maxReach or c <= 0):
                        reach = False
                    cords = [point[0]-norms[index-1][0]*c,point[1]-norms[index-1][1]*c]
                    checkp = tree.getNearR(cords,[cords[0]*10000,cords[1]*9000])
                    
                  
                    if not(checkp == point):
                        A = [point[0] - cords[0],point[1] - cords[1]]
                        B = [point[0] - checkp[0],point[1] - checkp[1]]
                        theta = np.arccos((A[0] * B[0] + A[1] * B[1]) / (np.sqrt(np.power(np.abs(A[0]),2) + np.power(np.abs(A[1]),2)) * np.sqrt(np.power(np.abs(B[0]),2) + np.power(np.abs(B[1]),2))))
                        if(theta < 0.2):
                            
                            midp = [(point[0] + checkp[0])/2,(point[1] + checkp[1])/2]
                            if(np.abs(midp[0] - centerp[len(centerp) - 1][0]) < closew * pointtol * threshDistance and np.abs(midp[1] - centerp[len(centerp) - 1][1]) < closew * pointtol * threshDistance):
                                if(index == level):
                                    print(centerp)
                                    print(tempr)
                                    q = 1
                                    ptx = []
                                    pty = []
                                    while q < len(centerp):
                                        ptx.append(centerp[q][0])
                                        pty.append(centerp[q][1])
                                        q = q + 1
                                    ptx.append(point[0])
                                    pty.append(point[1])
                                    plt.scatter(ptx, pty, zorder = 3)
                                    print('rip')
                                centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
                                finPoints.append(centerp[len(centerp)-1])
                                finR.append(tempr[leng])
                                thinPoints.append(point)
                            reach = False
                        
                    reachitt = reachitt + 1
                case = True
            #going back and fourth from two radii    
            if i >= 3:
                repeat, order = checkRepeat(tempr)
                if repeat:
                    print('bruh')
                    n = 0
                    p = 0
                    sml = 0.0
                    while p < order:
                        if p == 0:
                            sml = tempr[len(tempr) - (order)]
                        else:
                            tmp = tempr[len(tempr)-(order - p)]
                            if tmp < sml:
                                sml = tempr[len(tempr)-(order-p)]
                                n = len(tempr) - (order - p)
                        p = p + 1
                        
                        
                   
                    
                    reach = True
                    reachitt = 1
                    while reach:
                        c = np.power(10, reachitt) * threshDistance * np.power(cweight,reachitt)
                        if(reachitt > maxReach or c <= 0):
                            reach = False
                        cords = [point[0]-norms[index-1][0]*c,point[1]-norms[index-1][1]*c]
                        checkp = tree.getNearR(cords,[cords[0]*10000,cords[1]*9000])
                        if not(checkp == point):
                            A = [point[0] - cords[0],point[1] - cords[1]]
                            B = [point[0] - checkp[0],point[1] - checkp[1]]
                            theta = np.arccos((A[0] * B[0] + A[1] * B[1]) / (np.sqrt(np.power(np.abs(A[0]),2) + np.power(np.abs(A[1]),2)) * np.sqrt(np.power(np.abs(B[0]),2) + np.power(np.abs(B[1]),2))))
                            if(theta < 0.2):
                                midp = [(point[0] + checkp[0])/2,(point[1] + checkp[1])/2]
                                if(np.abs(midp[0] - centerp[len(centerp) - 1][0]) < closew * pointtol * threshDistance and np.abs(midp[1] - centerp[len(centerp) - 1][1]) < closew * pointtol * threshDistance):
                                   if(index == level):
                                       print(centerp)
                                       q = 1
                                       ptx = []
                                       pty = []
                                       while q < len(centerp):
                                           ptx.append(centerp[q][0])
                                           pty.append(centerp[q][1])
                                           q = q + 1
                                       plt.scatter(ptx, pty, zorder = 3)
                                   finR.append(sml)
                                   finPoints.append(centerp[n])
                                   thinPoints.append(point)
                                reach = False
                            
                        reachitt = reachitt + 1
                    case = True
                    
                
            i = i + 1
        
            
        
        
        #guess values
        if index  != len(points):
            if(len(finR) >=1):
                guessr = finR[len(finR)-1] * 100  
            else:
                guessr = 1000
        index = index + 1
        
    
        
    #Thins out data and returns correct points. 
        
    fin2Points , fin2R = thin2D(points,thinPoints, finPoints, finR, threshDistance)    
    return fin2Points,fin2R

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
    