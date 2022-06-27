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

    #Finally finds radius
    radius = np.abs(dist / (2 * np.cos(theta)))
    
    return radius

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
        theta = np.arccos(inside)
    return theta

def normalize(points : list) -> list:
    i = 0
    retpoints = []
    dim = len(points[0])
    while i < len(points):
        tempx = points[i][0]
        tempy = points[i][1]
        if dim == 2:
            normalize = 1/np.sqrt(np.power(tempx,2)+np.power(tempy,2))
            tempx = tempx * normalize
            tempy = tempy * normalize
            retpoints.append([tempx,tempy])
        else:
            tempz = points[i][2]
            normalize = 1/np.sqrt(np.power(tempx,2)+np.power(tempy,2)+np.power(tempz,2))
            tempx = tempx * normalize
            tempy = tempy * normalize
            tempz = tempz * normalize
            retpoints.append([tempx,tempy,tempz])
        i = i + 1
    return retpoints

def getDistance(point1, point2) -> float:
    if len(point1) == 2:
        return np.sqrt(pow(np.abs(point1[0]-point2[0]),2) + pow(np.abs(point1[1]-point2[1]),2))
    else:
        return np.sqrt(np.abs(pow(point1[0]-point2[0],2)) + np.abs(pow(point1[1]-point2[1],2))+np.abs(pow(point1[2]-point2[2],2)))
    