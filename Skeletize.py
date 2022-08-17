# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:22:45 2022

@author: Graham Garcia
"""

from random  import randint,shuffle
# from sys import float_repr_style
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits import mplot3d
import numpy as np
# import csv
# import scipy
# import pandas as pd


# @profile
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

# @profile
def getRadius(point1, point2 , norm) -> float:
    #First finds theta
    dim = len(point1)
    dist = getDistance(point1,point2)
    if dim == 2:
        #Next calculate the midpoint
        mvec = [point2[0] - point1[0],point2[1] - point1[1]]
        #Then find dot product of the norm and mvec
        dot = mvec[0]*-1*norm[0] + mvec[1]*-1*norm[1]
    else:
        #Next calculate the midpoint
        mvec = [point2[0] - point1[0],point2[1] - point1[1],point2[2] - point1[2]]
        #Then find dot product of the norm and mvec
        dot = mvec[0]*-1*norm[0] + mvec[1]*-1*norm[1] + mvec[2]*-1*norm[2]
    #Next finds theta
    theta = np.arccos(min(1,dot/dist))
    # print(dot,dist,point1,point2)
    #Finally finds radius
    radius = np.abs(dist / (2 * np.cos(theta)))
    
    return radius
# @profile
def getAngle(vec1 : list, vec2 : list,len1 : float, len2 : float) -> float:
    
    i = 0
    inside = 0
    while i < len(vec1):
        inside += (vec1[i] * vec2[i])
        i += 1
    if inside == 0:
        theta = np.pi / 2
    else:
        inside = inside / (len1 * len2)
        if inside > 1 or inside < -1:
            inside = np.round(inside,6) 
        theta = np.arccos(inside)
    return theta
# @profile
def normalize(points : list) -> list:
    i = 0
    pts = []
    while i < len(points):
        pts.append(points[i])
        i += 1
    i = 0
    retpoints = []
    dim = len(pts[0])
    while i < len(pts):
        tempx = pts[i][0]
        tempy = pts[i][1]
        if dim == 2:
            normalize = 1/((tempx * tempx + tempy * tempy) ** 0.5)
            tempx = tempx * normalize
            tempy = tempy * normalize
            retpoints.append([tempx,tempy])
        else:
            tempz = pts[i][2]
            normalize = 1/((tempx*tempx + tempy*tempy+tempz*tempz) ** 0.5)
            tempx = tempx * normalize
            tempy = tempy * normalize
            tempz = tempz * normalize
            retpoints.append([tempx,tempy,tempz])
        i = i + 1
    return retpoints
# @profile
def getDistance(point1, point2) -> float:
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    if len(point1) == 2:
        return (x ** 2 + y ** 2) ** 0.5
    else:
        z = point1[2] - point2[2]
        return (x ** 2 + y ** 2 + z ** 2) ** 0.5
# @profile
def getPoint(point : list, vec : list):
    ret = []
    i = 0
    while i < len(point):
        ret.append(point[i] + vec[i])
        i += 1
    return ret
    


def randPN(points : list,norms : list ):
    #This function is for randomizing data, for more closely related time..
    i = 0
    ret = []
    #First ties points together
    while i < len(points):
        ret.append([])
        ret[i].append(points[i])
        ret[i].append(norms[i])
        # print(ret[i])
        i += 1
    shuffle(ret)
    i = 0
    rp = []
    rn = []
    while i < len(ret):
        rp.append(ret[i][0])
        rn.append(ret[i][1])
        i += 1
    return rp,rn


def getDeviation(list1 : list, list2 : list):
    #is inputted two *NORMALIZED VECTORS*, and returns the unbaised difference in all the comparable values
    #We score the deviation from 0 -> 1
    #0 = Completely opposite
    #1 = Completely same
    dim = len(list1)
    i = 0
    dx = 0.0
    while i < dim:
        dx += abs(list1[i] - list2[i])
        i += 1
    if dim == 2:
        return (-(dx/4) + 1)
    else:
        return (-(dx/6) + 1)
        