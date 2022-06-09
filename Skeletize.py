# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:45 2022

@author: Graham Garcia
"""

from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import numpy as np
import csv
import scipy
import pandas as pd


from DataStructures import normalize2D, normalize3D, kdTree, getDistance2D, getDistance3D

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
    #First finds theta
    dist = getDistance2D(point1,point2)
    
    #Next calculate the midpoint
    mvec = [point2[0] - point1[0],point2[1] - point1[1]]
    
    #Then find dot product of the norm and mvec
    dot = mvec[0]*-1*norm[0] + mvec[1]*-1*norm[1]

    #Next finds theta
    theta = np.arccos(min(1,dot/dist))

    #Finally finds radius
    radius = np.abs(dist / (2 * np.cos(theta)))
    
    return radius

def getRadius3D(point1,point2,norm) -> float:
    return 0


def thin2D(opts : list, measured : list, finPts : list, finR : list, pointDis):
    pts = []
    animd = []
    j = 0
    #interface points
    while j < len(opts):
        pts.append(opts[j])
        j = j + 1
    
   
    Threshold = pointDis  # applies for how much varying is allowed between a real point / fake point
    
    
    thin1p = []
    thin1r = []
    #removing the repeated points, as they arent very important 
    print('Thinning...')
    i = 0
    while i < len(finPts):
        # print(i,'/',len(finPts) - 1)
        j = i + 1
        noRepeat = True
        while j < len(finPts):
            if np.abs(finR[i] - finR[j]) <= Threshold/100:
                if np.abs(finPts[i][0] - finPts[j][0]) <= Threshold/100:
                    if np.abs(finPts[i][1] - finPts[j][1]) <= Threshold/100:
                        noRepeat = False
                        animd.append(finPts[i])
                        
            j += 1
            
            
        if noRepeat:
            thin1p.append(finPts[i])
            thin1r.append(finR[i])
        i += 1
    
    
    thin2p = []
    thin2r = []
    #Next removes points that are too distant from any others
    tree = kdTree(thin1p, 2)
    tree2 = kdTree(pts,2)
    i = 0
    while i < len(thin1p):
        # print(i,'/',len(thin1p) - 1)
        if getDistance2D(thin1p[i],tree.getNearR(thin1p[i],thin1p[i])) < pointDis and getDistance2D(thin1p[i],tree2.getNearR(thin1p[i],[thin1p[i][0]*13042,thin1p[i][1]*-9827])) > pointDis :
            thin2p.append(thin1p[i])
            thin2r.append(thin1r[i])
        else:
            animd.append(thin1p[i])
        i += 1
        
    print(len(finPts),"-->",len(thin1p),"Through repeat removal")
    print(len(thin1p),"-->",len(thin2p),"Through distance")
    return thin1p,thin1r,animd
    
    
def Skeletize2D(points : list, norms : list,start : int, stop : int):
    print('Skeletizing...')
    #skeletize takes in 
    #points, the given plain list of points [x,y] for 2D case
    #norms, a list of not yet normalized normal points [n_x,n_y] here for 2D case
    
    #then returns 2 things
    # finPoints = [[x,y],...] of skeleton points
    # finR = [r1,r2,...] of the radius of each skeleton point

    countbreak = 0
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
    
    #Stored as lists within lists, each list is a points individual searches
    acp = []#CenterPoints saved when making an animation
    atp = []#TreePoints saved when making an animation
    ar = []#Radius saved when making animation
   
    ap = []#MainPoints saved when making animation
    an = []#Normals saved when making animation
    
    
    thinPoints = []
    finPoints = []#list of skeletized points
    finR = []#radius values of each point
    index = 1
    guessr = 0
    for point in points:
        
        tacp = []
        tatp = []
        tar = []
        
        
        if index - 1 >= start and index - 1 <= stop:
            ap.append(point)
            an.append(norms[index-1])
        
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
        # print(index, '/' , len(points))
        #solve loop
        while not case:
            centerp.append([float(point[0]-norms[index-1][0]*tempr[len(tempr)-1]),float(point[1]-norms[index-1][1]*tempr[len(tempr)-1])])
            testp = tree.getNearR(centerp[len(centerp)-1], point)
            tempr.append(np.round(getRadius2D(point,testp,norms[index-1]),8))
            leng = len(tempr)-1
            
            
            #INFORMATION CAPTURE FOR ANIMATION
            if index - 1 >= start and index - 1 <= stop:
                tacp.append(centerp[len(centerp) - 1])
                tatp.append(testp)
                tar.append(tempr[leng-1])
            
            
            #check for radius size
            if tempr[leng] < threshDistance and i > 1:
                finPoints.append(centerp[len(centerp) - 2])
                finR.append(tempr[leng - 2])
                thinPoints.append(point)
                
                if finPoints[len(finPoints) - 1][0] > 0.7 and finR[len(finR) - 1] < 0.2:
                    print('1')
	 
                tacp.append(centerp[len(centerp) - 2])
                tatp.append(tatp[len(tatp) - 2])
                tar.append(tempr[leng-2])
                
                case = True
                countbreak += 1
                break
            
            #for if within the threshold distance of the interface(too close to surface to be reasonable)
            #if getDistance2D(point, testp) < tempr[leng] and i > 3:
            #    finPoints.append(centerp[len(centerp) - 2])
            #    finR.append(tempr[leng - 1])
		    #    
            #    if finPoints[len(finPoints) - 1][0] > 0.7 and finR[len(finR) - 1] < 0.2:
            #        print('2')
            #    
            #    tacp.append(centerp[len(centerp) - 2])
            #    tatp.append(tatp[len(tatp) - 2])
            #    tar.append(tempr[leng-1])
            #    
            #    thinPoints.append(point)
            #    case = True
            #    countbreak += 1
            #    break
            
            
            
            #cases for cacthing when stuck  
            if np.abs(tempr[leng] - tempr[leng-1]) < 0.00001 and i > 1:
                if getDistance2D(point, testp) < tempr[leng]:
                    finPoints.append(centerp[len(centerp) - 2])
                    finR.append(tempr[leng - 1])
                    if finPoints[len(finPoints) - 1][0] > 0.7 and finR[len(finR) - 1] < 0.2:
                        print('3')
                else:
                    finPoints.append(centerp[len(centerp)-1])
                    finR.append(tempr[leng])
                    if finPoints[len(finPoints) - 1][0] > 0.7 and finR[len(finR) - 1] < 0.2:
                        print('4')
                thinPoints.append(point)
                case = True
                break
            
            
            #going back and fourth from two radii    
            if i >= 3:
                repeat, order = checkRepeat(tempr)
                if repeat:
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
                        
                    finR.append(sml)
                    finPoints.append(centerp[n])
                    thinPoints.append(point)
                    case = True
                    
                
            i = i + 1
            
            
            
        #Adding each itteration's variable
        acp.append(tacp)
        atp.append(tatp)
        ar.append(tar)
        
        
        #guess values
        if index  != len(points):
            if(len(finR) >=1):
                guessr = getDistance2D(point,testp)*100
                if guessr < threshDistance * len(points) / 2:
                    guessr = threshDistance * len(points) / 2
            else:
                guessr = 1000
        index = index + 1
        
    
    print('Skeletized', len(finPoints))
    animd = []
    anim = []#return for animation info
    #Thins out data and returns correct points. 
    tx = []
    ty = []
    ttx = []
    tty = []
    i = 0
    while i < len(points):
        tx.append(points[i][0])
        ty.append(points[i][1])
        i += 1
    i = 0
    while i < len(finPoints):
        ttx.append(finPoints[i][0])
        tty.append(finPoints[i][1])
        i += 1
    plt.scatter(tx,ty)
    plt.scatter(ttx,tty)
    plt.title('Before Removal')
    plt.savefig('BeforeThin.png')
    fin2Points , fin2R, animd = thin2D(points,thinPoints, finPoints, finR, threshDistance)    
    
    anim.append(acp)
    anim.append(atp)
    anim.append(ar)
    anim.append(ap)
    anim.append(an)
    # print('points with distance less than radius',countbreak)
    return fin2Points,fin2R, anim, animd

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
    
