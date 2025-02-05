# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:46:08 2022

@author: graha
"""
# from random  import randint
# from sys import float_repr_style
# import matplotlib
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits import mplot3d
import numpy as np
import time
# import csv
# import scipy
# import pandas as pd
# import numpy as np
import os
import sys
source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.insert(0,source)
from Skeletize import getDistance, getAngle, getDeviation, normalize

def quicksort(points : list = [], dimension : int = 0):
    return quicksortrunner(points,dimension,0,len(points)-1)

def quicksortrunner(points : list , dimension : int , first : int , last : int ,depth : int = 0):
    # print('sorting',depth)
    if first<last:
        splitpoint = partition(points,dimension,first,last)
        quicksortrunner(points, dimension, first, splitpoint - 1,depth=depth+1)
        quicksortrunner(points, dimension, splitpoint + 1,last,depth=depth+1)
        return points
def partition(points : list , dimension : int , first : int , last : int):
    if isinstance(points[0],SkelePoint):
        pivot = points[first].getAxis(dimension)
    else:
        pivot = points[first][dimension]
    left = first + 1
    right = last
    done = False
    while not done:
        if isinstance(points[0],SkelePoint):
            while left <= right and points[left].getAxis(dimension) <= pivot:
                left += 1
            while right >= left and points[right].getAxis(dimension) >= pivot:
                right -= 1
            if right < left:
                done = True
            else:
                temp = points[left]
                points[left] = points[right]
                points[right] = temp
        else:
            while left <= right and points[left][dimension] <= pivot:
                left += 1
            while right >= left and points[right][dimension] >= pivot:
                right -= 1
            if right < left:
                done = True
            else:
                temp = points[left]
                points[left] = points[right]
                points[right] = temp
    temp = points[first]
    points[first] = points[right]
    points[right] = temp
    return right





class kdTree:
    def __init__(self , points : list , depth : int = 0 , * , rads : list = [],dimensions : int = 0,cpuavail : int = 1,limit:int=0):
        # print('making {}'.format(depth))
        #first converts to skeleton points
        if not(dimensions == 0):
            self.dimensions = dimensions
        if depth == 0:
            t = len(points) / 100
            t = t ** (1/3)
            t = np.ceil(t*4)
            limit = int(t)
        if depth == 0 and not(isinstance(points[0],SkelePoint)):
            st = time.time()
            print('Initiating k-d,Tree')
            self.dimensions = len(points[0])
            i = 0
            tp = []
            while i < len(points):
                if len(rads) > 0:
                    tp.append(SkelePoint(points[i], rad=rads[i]))
                else:
                    tp.append(SkelePoint(points[i]))
                i += 1
            points = tp
            # ett = time.time()
            # ttt = ett - st
            # print('point conversion took {} minuites and {} seconds to make {} SkelePoints'.format(ttt // 60,ttt % 60, len(points)))
        self.depth = depth
        #Next is the actual process of creating the struct
        if len(points) > limit:
            #print(limit)
            self.split = True
            self.axis = depth % self.dimensions
            points = quicksort(points,self.axis)
            # ttt = time.time() -  stt
            # print('quicksearch took {} minuites and {} seconds to sort {} SkelePoints'.format(ttt // 60,ttt % 60, len(points)))
            mid = len(points) // 2
            self.node = points[mid]
            i = 0
            pointsl = []
            while i < mid:
                pointsl.append(points[i])
                i += 1
            self.leafL = kdTree(pointsl,depth + 1,dimensions=self.dimensions,limit=limit)
            i = mid + 1
            pointsr = []
            while i < len(points):
                pointsr.append(points[i])
                i += 1
            self.leafR = kdTree(pointsr,depth + 1,dimensions=self.dimensions,limit=limit)
        else:
            self.split = False
            i = 0
            self.points = []
            while i < len(points):
                self.points.append(points[i])
                i += 1
        if depth == 0:
            et = time.time()
            tt = et - st
            print('k-d tree took {} minuites and {} seconds to make'.format(tt // 60,tt % 60))
    def getNearR(self,inputdat : list):
        if len(inputdat) > 2:
            getRads = inputdat[2]
            if len(inputdat) > 3:
                cpuavail = inputdat[3]
            tdat = []
            tdat.append(inputdat[0])
            tdat.append(inputdat[1])
            inputdat = tdat
        else:
            getRads = False
        #print('searching...')
        dmin = 100000
        if self.split:
            pmin = self.node
            #If needs to go further
            if inputdat[0][self.axis] <= self.node.getAxis(self.axis):
                tpmin,dmin =  self.leafR.getNearR(inputdat)
                if tpmin != 0:
                    pmin = tpmin
                #Next checks node point and sees if its closer, overwrites if so
                if not(self.node.getPoint() == inputdat[0]) and not(self.node.getPoint() == inputdat[1]):
                    ndis = getDistance(inputdat[0], self.node.getPoint())
                    if ndis < dmin:
                        pmin = self.node
                        dmin =  ndis
                tdis = self.node.getAxis(self.axis) - inputdat[0][self.axis]
                if tdis <= dmin:
                    pmin1,dmin1 =  self.leafL.getNearR(inputdat)
                    if not(pmin1 == 0) and not(pmin1.getPoint() == inputdat[0]) and not(pmin1.getPoint() == inputdat[1]):
                        if dmin1 < dmin:
                            dmin = dmin1
                            pmin = pmin1
            else:
                tpmin,dmin =  self.leafL.getNearR(inputdat)
                if tpmin != 0:
                    pmin = tpmin
                #Next checks node point and sees if its closer, overwrites if so
                if not(self.node.getPoint() == inputdat[0]) and not(self.node.getPoint() == inputdat[1]):
                    ndis = getDistance(inputdat[0], self.node.getPoint())
                    if ndis < dmin:
                        pmin = self.node
                        dmin =  ndis
                tdis = inputdat[0][self.axis] - self.node.getAxis(self.axis)
                if tdis <= dmin:
                    pmin1,dmin1 =  self.leafR.getNearR(inputdat)
                    if not(pmin1 == 0) and not(pmin1.getPoint() == inputdat[0]) and not(pmin1.getPoint() == inputdat[1]):
                        if dmin1 < dmin and not(dmin1) == 0:
                            dmin = dmin1
                            pmin = pmin1


        else:
            #If at the lowest
            i = 0
            while i < len(self.points):
                if i == 0:
                    if not(self.points[i].getPoint() == inputdat[1]) and not(self.points[i].getPoint() == inputdat[0]):
                        dmin = getDistance(inputdat[0],self.points[i].getPoint())
                        pmin = self.points[i]
                else:
                   if not(self.points[i].getPoint() == inputdat[1]) and not(self.points[i].getPoint() == inputdat[0]):
                       tmin = getDistance(inputdat[0],self.points[i].getPoint())
                       if tmin < dmin:
                           dmin = tmin
                           pmin = self.points[i]
                i += 1
            if not('pmin' in locals()):
                pmin = 0
        if self.depth == 0:
            if getRads:
                return pmin.getPoint(),pmin.getRad()
            else:
                return pmin.getPoint()
        else:
            return pmin,dmin
    # def getVectorR(self,inputdat : list,*,depth : int = 0,scan = np.pi / 4,cpuavail : int = 1):
    #         #Get vector will get the closest n number of points to the search point
    #         #it consideres a 'scan' degree area along the given vector, will go deepest first as thats where the closest points should be
    #         #however it will store node value and compare on the way out if a given node is a better point than something returned

    #         if depth == 0:
    #             if len(inputdat) > 3:
    #                 getRads = inputdat[3]
    #                 if len(inputdat) > 4:
    #                     scan = inputdat[4]
    #                     if len(inputdat) > 5:
    #                         cpuavail = inputdat[5]
    #             tin = []
    #             tin.append(inputdat[0])
    #             tin.append([])
    #             i = 0
    #             while i < len(inputdat[1]):
    #                 tin[1].append(inputdat[1][i])
    #                 i += 1
    #             tin.append(inputdat[2])
    #             inputdat = tin

    #         retPoints = []
    #         retDist = []
    #         axis = depth % self.dimensions
    #         steps = ''
    #         if not(self.split):
    #             i = 0
    #             while i < len(self.points):
    #                 #each point it first checks the angle between the vectors.
    #                 j = 0
    #                 cvec = []
    #                 tpoint = []
    #                 while j < len(inputdat[1]):
    #                     cvec.append(self.points[i].getAxis(j) - inputdat[0][j])
    #                     tpoint.append(inputdat[0][j] + inputdat[1][j])
    #                     j += 1
    #                 tdis = getDistance(inputdat[0],self.points[i].getPoint())
    #                 theta = getAngle(inputdat[1],cvec,getDistance(inputdat[0],tpoint),tdis)
    #                 if theta  < (scan / 2):
    #                     #Within vector distance
    #                     if len(retPoints) < inputdat[2]:
    #                         retPoints.append(self.points[i])
    #                         retDist.append(tdis)
    #                     else:
    #                         j = 0
    #                         tagj = 0
    #                         mtdis = -1
    #                         while j < len(retPoints):
    #                             if tdis < retDist[j] and retDist[j] > mtdis:
    #                                 tagj = j
    #                                 mtdis = retDist[j]
    #                             j += 1
    #                         if not(mtdis == -1):
    #                             retPoints[tagj] = self.points[i]
    #                             retDist[tagj] = tdis
    #                 i += 1
    #         else:
    #             node = self.node
    #             #First want to see if the vector only reaches a specific leaf of the tree
    #             i = 0
    #             # vecax = []
    #             tp1 = []
    #             # tp2 = []
    #             nvec = []
    #             while i < len(inputdat[0]):
    #                 # if i == axis:
    #                 #     vecax.append(1)
    #                 # else:
    #                 #     vecax.append(0)
    #                 tp1.append(inputdat[0][i] + inputdat[1][i])
    #                 # tp2.append(inputdat[0][i] + vecax[i])
    #                 nvec.append(node.getAxis(i) - inputdat[0][i])
    #                 i += 1
    #             # theta = getAngle(inputdat[1],vecax,getDistance(inputdat[0],tp1),getDistance(inputdat[0],tp2))
    #             # mat = []
    #             # if self.dimensions == 2:
    #             #     if axis == 0:
    #             #         mat.append(inputdat[1][0] * np.cos(scan / 2) - inputdat[1][1] * np.sin(scan / 2))
    #             #         mat.append(inputdat[1][0] * np.cos(-scan / 2) - inputdat[1][1] * np.sin(-scan / 2))
    #             #     else:
    #             #         mat.append(inputdat[1][0] * np.sin(scan / 2) + inputdat[1][1] * np.cos(scan / 2))
    #             #         mat.append(inputdat[1][0] * np.sin(-scan / 2) +inputdat[1][1] * np.cos(-scan / 2))
    #             # else:
    #             #     if axis == 0:
    #             #         #X axis rotation
    #             #         mat.append([])
    #             #         mat[0].append(inputdat[1][0] * np.cos(scan / 2) - inputdat[1][1] * np.sin(scan / 2))
    #             #         mat[0].append(inputdat[1][0] * np.cos(-scan / 2) - inputdat[1][1]  * np.sin(-scan / 2))
    #             #         mat.append([])
    #             #         mat[1].append(inputdat[1][0] * np.sin(scan / 2) + inputdat[1][1] * np.cos(scan / 2))
    #             #         mat[1].append(inputdat[1][0] * np.sin(-scan / 2) + inputdat[1][1] * np.cos(-scan / 2))
    #             #         mat.append([])
    #             #         mat[2].append(inputdat[1][2])
    #             #         mat[2].append(inputdat[1][2])
    #             #     elif axis == 1:
    #             #         #Y axis rotation
    #             #         mat.append([])
    #             #         mat[0].append(inputdat[1][0] * np.cos(scan / 2) + inputdat[1][2] * np.sin(scan / 2))
    #             #         mat[0].append(inputdat[1][0] * np.cos(-scan / 2) + inputdat[1][2]  * np.sin(-scan / 2))
    #             #         mat.append([])
    #             #         mat[1].append(inputdat[1][1])
    #             #         mat[1].append(inputdat[1][1])
    #             #         mat.append([])
    #             #         mat[2].append(-inputdat[1][0] * np.sin(scan / 2) + inputdat[1][2] * np.cos(scan / 2))
    #             #         mat[2].append(-inputdat[1][0] * np.sin(-scan / 2) + inputdat[1][2] * np.cos(-scan / 2))
    #             #     else:
    #             #         #Z axis rotation
    #             #         mat.append([])
    #             #         mat[0].append(inputdat[1][0])
    #             #         mat[0].append(inputdat[1][0])
    #             #         mat.append([])
    #             #         mat[1].append(inputdat[1][1] * np.cos(scan / 2) - inputdat[1][2] * np.sin(scan / 2))
    #             #         mat[1].append(inputdat[1][1] * np.cos(-scan / 2) - inputdat[1][2] * np.sin(-scan / 2))
    #             #         mat.append([])
    #             #         mat[2].append(inputdat[1][1] * np.sin(scan / 2) + inputdat[1][2] * np.cos(scan / 2))
    #             #         mat[2].append(inputdat[1][1] * np.sin(-scan / 2) + inputdat[1][2] * np.cos(-scan / 2))
    #             # if depth == 0:
    #             #     print()
    #             #     print(mat,theta)
    #             #     print(inputdat[0],inputdat[1])


    #             if node.getAxis(axis) < inputdat[0][axis] and inputdat[1][axis] > 0.1:
    #                 retPoints,retDist,steps = self.leafR.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
    #                 if depth < 4 and len(retPoints) == 0:
    #                     if depth == 0:
    #                         print()
    #                         print('went left and shouldnt of ?',depth)
    #                         print('point',inputdat[0],'norm',inputdat[1])
    #                         print('node',node.getPoint())
    #                         print('axis',axis)
    #                         print('steps:',steps)
    #                         print()
    #                     else:
    #                         steps += 'went left and shouldnt of?'
    #                         steps += str(node.getPoint())
    #                         steps += str(depth)

    #             elif node.getAxis(axis) > inputdat[0][axis] and inputdat[1][axis] < 0.1:
    #                 retPoints,retDist,steps = self.leafL.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
    #                 if depth < 4 and len(retPoints) == 0:
    #                     if depth == 0:
    #                         print()
    #                         print('went right and shouldnt of ?',depth)
    #                         print('point',inputdat[0],'norm',inputdat[1])
    #                         print('node',node.getPoint())
    #                         print('axis',axis)
    #                         print(steps)
    #                         print()
    #                     else:
    #                         steps += 'went right and shouldnt of ?'
    #                         steps += str(node.getPoint())
    #                         steps += str(depth)
    #             else:
    #                 # print('bruh',mat[axis])
    #                 tretPoints = []
    #                 tretDist = []
    #                 retPointsl = []
    #                 retPointsr = []
    #                 retDistl = []
    #                 retDistr = []
    #                 retPointsl,retDistl,stepsl = self.leafL.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
    #                 retPointsr,retDistr,stepsr = self.leafR.getVectorR(inputdat,depth = depth + 1, scan = scan,cpuavail=cpuavail)
    #                 steps += stepsl
    #                 steps += stepsr
    #                 if depth < 4 and len(retPointsl) == 0 and len(retDistr) == 0:
    #                     if depth == 0:
    #                         print()
    #                         print('went Both and got nothin?',depth)
    #                         print('point',inputdat[0],'norm',inputdat[1])
    #                         print('node',node.getPoint())
    #                         print('axis',axis)
    #                         print(steps)
    #                         print()
    #                     else:
    #                         steps += 'went Both and got nothin?'
    #                         steps += str(node.getPoint())
    #                         steps += str(depth)

    #                 i = 0
    #                 while i < len(retPointsl):
    #                     tretPoints.append(retPointsl[i])
    #                     tretDist.append(retDistl[i])
    #                     i += 1
    #                 i = 0
    #                 while i < len(retPointsr):
    #                     tretPoints.append(retPointsr[i])
    #                     tretDist.append(retDistr[i])
    #                     i += 1
    #                 ntheta = getAngle(inputdat[1],nvec,getDistance(inputdat[0],tp1),getDistance(inputdat[0],node.getPoint()))
    #                 #gets the node point if it falls inside the criteria
    #                 if ntheta < (scan / 2):
    #                     tretPoints.append(node)
    #                     tretDist.append(getDistance(inputdat[0],node.getPoint()))
    #                 #Finally aquires the best n# of points from big list
    #                 i = 0
    #                 while i < len(tretPoints):
    #                     if i < inputdat[2]:
    #                         retPoints.append(tretPoints[i])
    #                         retDist.append(tretDist[i])
    #                     else:
    #                         tdis = tretDist[i]
    #                         j = 0
    #                         tagj = 0
    #                         mtdis = -1
    #                         while j < len(retPoints):

    #                             if tdis < retDist[j] and retDist[j] > mtdis:
    #                                 tagj = j
    #                                 mtdis = retDist[j]
    #                             j += 1
    #                         if not(mtdis == -1):
    #                             retPoints[tagj] = tretPoints[i]
    #                             retDist[tagj] = tdis
    #                     i += 1

    #         if depth == 0:
    #             j = 0
    #             trp = []
    #             trad = []
    #             while j < len(retPoints):
    #                 trp.append(retPoints[j].getPoint())
    #                 if getRads:
    #                     trad.append(retPoints[j].getRad())
    #                 j += 1
    #             if getRads:
    #                 return trp, trad
    #             else:
    #                 return trp
    #         else:
    #             return retPoints,retDist,steps


    def getVectorR(self, data : list):
        # print('searching',data[3])
        if len(data) >= 5:
            tdata = []
            tdata.append(data[0])
            tdata.append(data[1])
            tempthresh = data[2] * 10
            tdata.append(tempthresh)
            tdata.append(data[3])
            data = tdata
        # if data[3] == 0:
        #     if len(data) == 5:
        #         getRads = data[4]
        #         data = data[0:4]

        #Finds the differnce between the given vector(from root point) and from the root point to the search point.
        #Comapres, if close enough, the the point should be close enough to the other side of the shape.
        retpts = []
        retdev = []
        if not(self.split):
            # print('searching {}'.format(data[3]))
           # i = 0
           # while i < len(self.points):
           #     tvec = []
           #     j = 0
           #     while j < self.points[i].dimensions:
           #         tvec.append(self.points[i].getAxis(j) - point[j])
           #         j += 1
           #     tvec = normalize([tvec])
           #     dev = getDeviation(tvec[0],vector)
           #     if len(retpts) < n:
           #         retpts.append(self.points[i])
           #         retdev.append(dev)
           #     else:
           #         q = 0
           #         ind = -1
           #         tlow = 1.1
           #         while q < len(retpts):
           #             if retdev[q] < tlow:
           #                 tlow =  retdev[q]
           #                 ind = q
           #             q += 1
           #         if retdev[ind] < dev:
           #             retdev[ind] = dev
           #             retpts[ind] = self.points[i]
           #     i += 1
            i = 0
            low = 0
            ind = 0
            while i < len(self.points):
                if self.points[i].getPoint() != data[0]:
                    tvec = []
                    j = 0
                    while j < self.points[i].dimensions:
                        tvec.append(self.points[i].getAxis(j) - data[0][j])
                        j += 1
                    dev = getDeviation(tvec,data[1])
                    if dev > low:
                        low = dev
                        ind = i
                i += 1
            #print(low)
            retpts.append(self.points[ind])
            retdev.append(low)
            # print(retpts,retdev,data[3])
        else:
            node = self.node
            #How to determine which nodes to search through.
            if node.getAxis(self.axis) < data[0][self.axis] and data[1][self.axis] > 0:
                #if want right leaf only
                tdata = data.copy()
                tdata[3] += 1
                retpts,retdev = self.leafR.getVectorR(tdata)
                #   print('r only')
                # print(node.getPoint(),'along the',self.axis)
            elif node.getAxis(self.axis) > data[0][self.axis] and data[1][self.axis] < 0:
                #if want left leaf only
                tdata = data.copy()
                tdata[3] += 1
                retpts,retdev = self.leafL.getVectorR(tdata)
                # print('l only')
                # print(node.getPoint(),'along the',self.axis)
            else:
                trdata = data.copy()
                trdata[3] += 1
                tldata = data.copy()
                tldata[3] += 1
                retptsl,retdevl =  self.leafL.getVectorR(tldata)
                retptsr,retdevr =  self.leafR.getVectorR(trdata)
                # print('both')
                # print(depth,data[0],data[1])
                # print('left option',retptsl[0].getPoint(),retdevl[0])
                # print('right option',retptsr[0].getPoint(),retdevr[0])
                dl = getDistance(data[0],retptsl[0].getPoint())
                dr = getDistance(data[0],retptsr[0].getPoint())
                #TEST3
                #print()
                #print('point',data[0],data[3])
                diffdev = abs(retdevl[0]-retdevr[0])
                #if data[0][0] < 0.18:
                #    if dl < data[2]/10 or dr < data[2]/10:
                #        print()
                #        print('Point:',data[0],data[3],'L-R')
                #        print('distance',dl,dr)
                #        print('dev',retdevl[0],retdevr[0])
                #        print('points',retptsl[0].getPoint(),retptsr[0].getPoint())
                #        print()

                #TEST4
                if len(retdevr) == 0 or len(retptsr) == 0:
                    retpts = retptsl
                    retdev = retdevl
                elif len(retdevl) == 0 or len(retptsl) == 0:
                    retpts = retptsr
                    retdev = retdevr
                #LOGIC FOR THIN BAG, theres only 2 surfaces realistically, so prio further
                #if (dl > data[2]*10 and dr < data[2]*10) or (dl < data[2]*10 and dr > data[2]*10):
                #    if retdevr[0] > 0.9 and retdevl[0] > 0.9:
                #        if dr > dl:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            retpts = retptsl
                #            retdev = retdevl
                #    else:
                #        if retdevr[0] > retdevl[0]:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            retpts = retptsl
                #            retdev = retdevl
                #elif dl > data[2]*10 and dr > data[2]*10:
                #    if retdevr[0] > retdevl[0]:
                #        retpts = retptsr
                #        retdev = retdevr
                #    else:
                #        retpts = retptsl
                #        retdev = retdevl
                #else:

                #3D LOGIC
                #if retdevr[0] > 0.9 and retdevl[0] > 0.9:
                #    if dr < data[2] and dl < data[2]:
                #        if dr > dl and retdevr[0] > retdevl[0]:
                #            retpts = retptsr
                #            retdev = retdevr
                #        elif dl > dr and retdevl[0] > retdevr[0]:
                #            retpts = retptsl
                #            retdev = retdevl
                #        else:
                #            #Here, Both are pretty close to on par with normal, but none is a decisive winner.
                #            #So we choose off of retdev for maximazation, and theyre very close anyways
                #            if retdevr[0] > retdevl[0]:
                #                retpts = retptsr
                #                retdev = retdevr
                #            else:
                #                retpts = retptsl
                #                retdev = retdevl

                #    elif dr > data[2] and dl > data[2]:
                #        if dr < dl:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            retpts = retptsl
                #            retdev = retdevl
                #    else:
                #        #This should be the part to switch if next run doesnt work.
                #        if dr > dl:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            retpts = retptsl
                #            retdev = retdevl
                #else:
                #    if retdevr[0] > retdevl[0]:
                #        retpts = retptsr
                #        retdev = retdevr
                #    else:
                #        retpts = retptsl
                #        retdev = retdevl

                #LOGIC, WORKS For 2D Spiral, Taking a differnet approach here.
                elif diffdev < data[2]*10:
                    if dl < data[2]/10 and dr < data[2]/10:
                        if dl < dr:
                            retpts = retptsl
                            retdev = retdevl
                        else:
                            retpts = retptsr
                            retdev = retdevr
                    else:
                        if dl < dr:
                            retpts = retptsl
                            retdev = retdevl
                        else:
                            retpts = retptsr
                            retdev = retdevr
                else:
                    #print(diffdev,data[2])
                    #if retdevl[0] > retdevr[0]:
                    #    retpts = retptsl
                    #    retdev = retdevl
                    #else:
                    #    retpts = retptsr
                    #    retdev = retdevr

                    if dl > data[2]/10 and dr > data[2]/10:
                        if retdevl[0] > retdevr[0]:
                            retpts = retptsl
                            retdev = retdevl
                        else:
                            retpts = retptsr
                            retdev = retdevr
                    elif dl < data[2]/10 and dr < data[2]/10:
                        if retdevl[0] > retdevr[0]:
                            retpts = retptsl
                            retdev = retdevl
                        else:
                            retpts = retptsr
                            retdev = retdevr
                    else:
                        if retdevl[0] > retdevr[0] and dl < dr:
                            retpts = retptsl
                            retdev = retdevl
                        elif retdevl[0] < retdevr[0] and dl < dr:
                            retpts = retptsr
                            retdev = retdevr
                        else:
                            if dl > dr:
                                retpts = retptsl
                                retdev = retdevl
                            else:
                                retpts = retptsr
                                retdev = retdevr
            #Finally we need to test if the node is a better substitution instead of the
            if not(node.getPoint() == data[0]):
                j = 0
                tvec = []
                while j < node.dimensions:
                    tvec.append(node.getAxis(j) - data[0][j])
                    j += 1
                dev = getDeviation(tvec,data[1])
                diffdev = abs(dev-retdev[0])
                dn = getDistance(data[0],node.getPoint())
                dl = getDistance(data[0],retpts[0].getPoint())

                #thin bag
                #if (dn > data[2]*10 and dl < data[2]*10) or (dn < data[2]*10 and dl > data[2]*10):
                #    if dev > 0.9 and retdev[0] > 0.9:
                #        if dn > dl:
                #            retpts = [node]
                #            retdev = [dev]
                #    else:
                #        if dev > retdev[0]:
                #            retpts = [node]
                #            retdev = [dev]
                #elif dn > data[2]*10 and dl > data[2]*10:
                #    if dev > retdev[0]:
                #        retpts = [node]
                #        retdev = [dev]
                #else:
                if dev > 0.9 and retdev[0] > 0.9:
                    if dn < data[2] and dl < data[2]:
                        if dn > dl and dev > retdev[0]:
                            retpts = [node]
                            retdev = [dev]
                        else:
                            if dev > retdev[0]:
                                retpts = [node]
                                retdev = [dev]
                    elif dn > data[2] and dl > data[2]:
                        if dn < dl:
                            retpts = [node]
                            retdev = [dev]
                    else:
                        if dn > dl:
                            retpts = [node]
                            retdev = [dev]

                else:
                    if dev > retdev[0]:
                        retpts = [node]
                        retdev = [dev]

                ###2D Spiral Case
                ###if diffdev < data[2]*10:
                ###    if dl < data[2]/10 and dn < data[2]/10:
                ###        if dl < dn:
                ###            retpts = [node]
                ###            retdev = [dev]
                ###    else:
                ###        if dl > dn:
                ###            retpts = [node]
                ###            retdev = [dev]
                ###else:
                ###    if retdev[0] < dev:
                ###        retpts = [node]
                ###        retdev = [dev]
                    #print(diffdev)
                    #if dl > data[2]/10 and dn > data[2]/10:
                    #    if retdev[0] < dev:
                    #        retpts = [node]
                    #        retdev = [dev]
                    #elif dl < data[2]/10 and dn < data[2]/10:
                    #    if retdev[0] < dev:
                    #        retpts = [node]
                    #        retdev = [dev]
                    #else:
                    #    if retdev[0] < dev and dl < dn:
                    #        retpts = [node]
                    #        retdev = [dev]




                #TEST 3
                #if diffdev < data[2]:
                #    if dl > data[2]/10 and dr > data[2]/10:
                #        if dl > dr:
                #            retpts = retptsr
                #            retdev = retdevr
                #        elif dl < dr:
                #            retpts = retptsl
                #            retdev = retdevl
                #        else:
                #            if dl < dr:
                #                retpts = retptsl
                #                retdev = retdevl
                #            else:
                #                retpts = retptsr
                #                retdev = retdevr

                #    else:
                #        #Here we have the potential of having a winner/Final Point. So we want to be more specific
                #        if dl < data[2]/10 and dr < data[2]/ 10:
                #            #if both are small, then we choose the more on one
                #            if retdevr[0] > retdevl[0]:
                #                retpts = retptsr
                #                retdev = retdevr
                #            else:
                #                retpts = retptsl
                #                retdev = retdevl
                #        elif dl < data[2]/10 and retdevl[0] > 0.8:
                #            retpts = retptsl
                #            retdev = retdevl
                #        elif dr < data[2]/10 and retdevr[0] > 0.8:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            if dl > dr:
                #                retpts = retptsl
                #                retdev = retdevl
                #            else:
                #                retpts = retptsr
                #                retdev = retdevr
                #else:
                #    #All results here will be within 0.05 dev of eachother. This is where we consider which ones are good enough to thresh and which ones arent
                #    if dl < data[2]/10 and dr < data[2]/ 10:
                #        #both small and not close to eachother, so take the most on
                #        print('b')
                #        if retdevr[0] > retdevl[0]:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            retpts = retptsl
                #            retdev = retdevl
                #    elif dl < data[2]/10 and retdevl[0] > 0.8:
                #        print('l')
                #        retpts = retptsl
                #        retdev = retdevl
                #    elif dr < data[2]/10 and retdevr[0] > 0.8:
                #        print('r')
                #        retpts = retptsr
                #        retdev = retdevr
                #    else:
                #        if retdevr[0] > retdevl[0]:
                #            retpts = retptsr
                #            retdev = retdevr
                #        else:
                #            retpts = retptsl
                #            retdev = retdevl


                #print()

                #TEST2
                # if dl > data[2] * 2 and dr > data[2] * 2:
                #     if dl > dr and retdevr[0] > 0.8:
                #         retpts = retptsr
                #         retdev = retdevr
                #         # print('went r')
                #     elif dl < dr and retdevl[0] > 0.8:
                #         retpts = retptsl
                #         retdev = retdevl
                #         # print('went l')
                #     else:
                #         #There isnt a clear winner of closer and more on. so we opt for more on now
                #         if retdevr[0] > retdevl[0]:
                #             retpts = retptsr
                #             retdev = retdevr
                #             # print('went r')
                #         else:
                #             retpts = retptsl
                #             retdev = retdevl
                #             # print('went l')
                # else:
                #     if dr > data[2]:
                #         retpts = retptsr
                #         retdev = retdevr
                #         # print('went r')
                #     elif dl > data[2]:
                #         retpts = retptsl
                #         retdev = retdevl
                #         # print('went l')
                #     else:
                #         #When both end up being 'toosmall'
                #         print('kinda small eh')
                #         if retdevr[0] > retdevl[0]:
                #             retpts = retptsr
                #             retdev = retdevr
                #             # print('went r')
                #         else:
                #             retpts = retptsl
                #             retdev = retdevl

                #TEST1
                # i = 0
                # while i < len(retptsl):
                #     if len(retpts) < n:
                #         retpts.append(retptsl[i])
                #         retdev.append(retdevl[i])
                #     else:
                #         j = 0
                #         ind = -1
                #         tlow = 1.1
                #         while j < len(retpts):
                #             if retdev[j] < tlow:
                #                 tlow = retdev[j]
                #                 ind = j
                #             j += 1
                #         if retdev[ind] < retdevl[i]:
                #             retdev[ind] = retdevl[i]
                #             retpts[ind] = retptsl[i]
                #     i += 1
                # i = 0
                # while i < len(retptsr):
                #     if len(retpts) < n:
                #         retpts.append(retptsr[i])
                #         retdev.append(retdevr[i])
                #     else:
                #         j = 0
                #         ind = -1
                #         tlow = 1.1
                #         while j < len(retpts):
                #             if retdev[j] < tlow:
                #                 tlow = retdev[j]
                #                 ind = j
                #             j += 1
                #         if retdev[ind] < retdevr[i]:
                #             retdev[ind] = retdevr[i]
                #             retpts[ind] = retptsr[i]
                #     i += 1

            # if len(retpts) < n:
            #     q = 0
            #     tv = 0
            #     while q < self.dim:
            #         tv.append(node.getAxis(q) - point[q])
            #         q += 1
            #     retpts.append(node)
            #     retdev.append(getDeviation(vector,tv))
        # if depth == 0:
        #     print('Chose the point',retpts[0].getPoint(),'for',data[0])
        # print(retpts,retdev)
        # print(self.split,data[3])
        return retpts,retdev


    # def getInR(self,point : list, dim : float, mode : int,depth : int = 0,*,getRads : bool = False):
    #      #Returns all the points which lie inside a give area arround a certain point
    #      #Mode 0 => Square area, point in center, side = 2 * dim
    #      #Mode 1 => Circle area, point in center, rad  = dim
    #      retPoints = []
    #      axis = depth % self.dimensions
    #      node = self.node
    #      if not(self.split):
    #          i = 1
    #          while i < len(tree):
    #              if mode == 0:
    #                  if tree[i][0] >= point[0] - dim and tree[i][0] <= point[0] + dim:
    #                      if tree[i][1] >= point[1] - dim and tree[i][1] <= point[1] + dim:
    #                          if self.dmensions == 2 or (self.dimensions == 3 and tree[i][2] >= point[2] - dim and tree[i][2] <= point[2] + dim):
    #                              retPoints.append(tree[i])
    #              else:
    #                  if getDistance(point,tree[i]) <= dim:
    #                      retPoints.append(tree[i])
    #              i += 1
    #      else:
    #          #Uses square to obtain all possible sections that might be needed
    #          #mode only comes into play when searching at bottom layers, however adding the node points
    #          #will still depend on mode
    #          pts = []
    #          rads = []
    #          if point[axis] - dim > node[axis]:
    #              if getRads:
    #                  pts,rads = self.getInR(point,dim,mode,tree[2],depth + 1, getRads=True,rtree = rtree[2])
    #              else:
    #                  pts = self.getInR(point,dim,mode,tree[2],depth + 1)
    #          elif point[axis] + dim < node[axis]:
    #              if getRads:
    #                  pts,rads = self.getInR(point,dim,mode,tree[1],depth + 1, getRads = True, rtree = rtree[1])
    #              else:
    #                  pts = self.getInR(point,dim,mode,tree[1],depth + 1)
    #          else:
    #              if getRads:
    #                  pts1,rad1 = self.getInR(point,dim,mode,tree[1],depth + 1, getRads = True, rtree = rtree[1])
    #                  pts2,rad2 = self.getInR(point,dim,mode,tree[2],depth + 1, getRads = True, rtree = rtree[2])
    #              else:
    #                  pts1 = self.getInR(point,dim,mode,tree[1],depth + 1)
    #                  pts2 = self.getInR(point,dim,mode,tree[2],depth + 1)
    #              i = 0
    #              while i < len(pts1):
    #                  pts.append(pts1[i])
    #                  if getRads:
    #                      rads.append(rad1[i])
    #                  i += 1
    #              i = 0
    #              while i < len(pts2):
    #                  pts.append(pts2[i])
    #                  if getRads:
    #                      rads.append(rad2[i])
    #                  i += 1
    #              #Note only needs to check node here
    #              if mode == 0:
    #                  if node[0] >= point[0] - dim and node[0] <= point[0] + dim:
    #                      if node[1] >= point[1] - dim and node[1] <= point[1] + dim:
    #                          if self.dmensions == 2 or (self.dimensions == 3 and node[2] >= point[2] - dim and node[2] <= point[2] + dim):
    #                              pts.append(node)
    #                              if getRads:
    #                                  rads.append(rnode)
    #              else:
    #                  if getDistance(point,node) <= dim:
    #                      pts.append(node)
    #                      if getRads:
    #                          rads.append(rnode)

    #          if len(pts) > 0:
    #              i = 0
    #              while i < len(pts):
    #                  retPoints.append(pts[i])
    #                  if getRads:
    #                      retR.append(rads[i])
    #                  i += 1
    #      if getRads:
    #          return retPoints,retR
    #      else:
    #          return retPoints

class SplitTree:
    #Split Tree is a versatile Quad/Oct tree designed for efficient stack storage for search
    def __init__(self,inpts,*,lastBox : list = [],inrad : list = [],dim : int = 0,dep : int = 0):
        #Bounds are stored in a node(center [x,y,z])
        self.count = len(inpts)
        self.state = False
        self.dep = dep
        if len(lastBox) != 0:
            self.lastBox = lastBox

        if not(dim == 2 or dim == 3):
            if isinstance(inpts[0],SkelePoint):
                self.dim = inpts[0].dimensions
            else:
                if len(inpts[0]) == 2:
                    self.dim = 2
                else:
                    self.dim = 3
        else:
            self.dim = dim
        if self.dim == 2:
            self.maxpts = 4
        else:
            self.maxpts = 8

        #Defining skele Points
        self.skelepts = []
        if not(len(inpts) == 0):
            if isinstance(inpts[0],list):
                i = 0
                while i < len(inpts):
                    if len(inrad) > 0:
                        self.skelepts.append(SkelePoint(inpts[i],rad = inrad[i]))
                    else:
                        self.skelepts.append(SkelePoint(inpts[i]))
                    i += 1
            else:
                i = 0
                while i < len(inpts):
                    self.skelepts.append(inpts[i])
                    i += 1
        #If there are too many points, will subdivide
        if len(self.skelepts) > self.maxpts:
            self.state = True
            self.subdivide()
        elif len(self.skelepts) > 0:
            self.getBox()

    def subdivide(self):
        #Creates the nodes for the new Quad/oct trees, and sorts points to their appropiate node
        self.leafs = []
        nodes = []
        points = []
        self.getBox()

        if self.dim == 2:
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            i = 0
            while i < len(self.skelepts):
                if self.skelepts[i].x > self.node[0] and self.skelepts[i].y > self.node[1]:
                    points[0].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y < self.node[1]:
                    points[1].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y > self.node[1]:
                    points[2].append(self.skelepts[i])
                else:
                    points[3].append(self.skelepts[i])
                i += 1
        else:
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            points.append([])
            i = 0
            while i < len(self.skelepts):
                if self.skelepts[i].x > self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[0].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z < self.node[2]:
                    points[1].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y < self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[2].append(self.skelepts[i])
                elif self.skelepts[i].x > self.node[0] and self.skelepts[i].y < self.node[1] and self.skelepts[i].z < self.node[2]:
                    points[3].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[4].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y > self.node[1] and self.skelepts[i].z < self.node[2]:
                    points[5].append(self.skelepts[i])
                elif self.skelepts[i].x < self.node[0] and self.skelepts[i].y < self.node[1] and self.skelepts[i].z > self.node[2]:
                    points[6].append(self.skelepts[i])
                else:
                    points[7].append(self.skelepts[i])
                i += 1
        i = 0
        while i < len(points):
            self.leafs.append(SplitTree(points[i],dim=self.dim,dep=self.dep + 1,lastBox = [self.node,self.c[0],self.c[1]]))
            i += 1
        self.skelepts = []
    def getBox(self):
        #First we find the local node
        if len(self.skelepts) > 0:
            avgx = 0
            avgy = 0
            avgz = 0
            if self.dep == 0:
                mx = [self.skelepts[0].x,self.skelepts[0].x]
                my = [self.skelepts[0].y,self.skelepts[0].y]
                if self.dim == 3:
                    mz = [self.skelepts[0].z,self.skelepts[0].z]
            for pt in self.skelepts:
                avgx += pt.x
                if self.dep == 0:
                    if pt.x > mx[0]:
                        mx[0] = pt.x
                    if pt.x < mx[1]:
                        mx[1] = pt.x
                avgy += pt.y
                if self.dep == 0:
                    if pt.y > my[0]:
                        my[0] = pt.y
                    if pt.y < my[1]:
                        my[1] = pt.y
                if self.dim == 3:
                    avgz += pt.z
                    if self.dep == 0:
                        if pt.z > mz[0]:
                            mz[0] = pt.z
                        if pt.z < mz[1]:
                            mz[1] = pt.z
            avgx = avgx / len(self.skelepts)
            avgy = avgy / len(self.skelepts)
            if self.dim == 3:
                avgz = avgz / len(self.skelepts)
                self.node = [avgx,avgy,avgz]
            else:
                self.node = [avgx,avgy]
        #Then We will get the bounding box
        if self.dep == 0:
            #If First layer, we want to define the corners
            if self.dim == 2:
                self.c = [[mx[0]*1.05,my[0]*1.05],[mx[1]*0.95,my[1]*0.95]]
            else:
                self.c = [[mx[0]*1.05,my[0]*1.05,mz[0]*1.05],[mx[1]*0.95,my[1]*0.95,mz[1]*0.95]]
        else:
            #If not first layer, we determine our current bounds using the last box's dimensions
            #Last box will come in as a [LastNode,c-max,c-min]
            self.c = [[],[]]
            for dim in range(self.dim):
                if self.node[dim] < self.lastBox[0][dim]:
                    self.c[0].append(self.lastBox[0][dim])
                    self.c[1].append(self.lastBox[2][dim])
                else:
                    self.c[0].append(self.lastBox[1][dim])
                    self.c[1].append(self.lastBox[0][dim])
        #We now have a node and bounding box for each and every layer we do, given we pass along the right information

    def addpoints(self, points : list = [],*,rads : list = [],spts : bool = False):
        #INPUTS points [skpt1,skpt2...] / [[x1,y1],[x2,y2],..]
        #Goes through all points and adds them to needed areas
        #First checks if the current leaf has been subdivided yet
        if len(points) == 0:
            return
        elif not(isinstance(points[0],SkelePoint)):
            i = 0
            tp = []
            while i < len(points):
                if len(rads) > 0:
                    tp.append(SkelePoint(points[i],rad=rads[i],connections=1))
                else:
                    tp.append(SkelePoint(points[i],connections=1))
                i += 1
            points = tp
        self.count += len(points)
        if not(self.state):#If final Leaf
            i = 0
            while i < len(points):
                self.skelepts.append(points[i])
                i += 1
            if len(self.skelepts) > self.maxpts:
                #If points more, than subdivide and add the point
                self.state = True
                self.subdivide()
        else:
            i = 0
            ne,se,nw,sw,a,b,c,d,e,f,g,h = [],[],[],[],[],[],[],[],[],[],[],[]
            while i < len(points):
                if self.dim == 2:
                    if points[i].x > self.node[0] and points[i].y > self.node[1]:
                        ne.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y < self.node[1]:
                        se.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y > self.node[1]:
                        nw.append(points[i])
                    else:
                        sw.append(points[i])
                else:
                    if points[i].x > self.node[0] and points[i].y > self.node[1] and points[i].z > self.node[2]:
                        a.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y > self.node[1] and points[i].z < self.node[2]:
                        b.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y < self.node[1] and points[i].z > self.node[2]:
                        c.append(points[i])
                    elif points[i].x > self.node[0] and points[i].y < self.node[1] and points[i].z < self.node[2]:
                        d.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y > self.node[1] and points[i].z > self.node[2]:
                        e.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y > self.node[1] and points[i].z < self.node[2]:
                        f.append(points[i])
                    elif points[i].x < self.node[0] and points[i].y < self.node[1] and points[i].z > self.node[2]:
                        g.append(points[i])
                    else:
                        h.append(points[i])
                i += 1
            if self.dim == 2:
                self.leafs[0].addpoints(ne)
                self.leafs[1].addpoints(se)
                self.leafs[2].addpoints(nw)
                self.leafs[3].addpoints(sw)
            else:
                self.leafs[0].addpoints(a)
                self.leafs[1].addpoints(b)
                self.leafs[2].addpoints(c)
                self.leafs[3].addpoints(d)
                self.leafs[4].addpoints(e)
                self.leafs[5].addpoints(f)
                self.leafs[6].addpoints(g)
                self.leafs[7].addpoints(h)


    def exists(self,point : list,tolerance : float,depth : int = 0):
        if self.state:
            #Go Deeper
            if self.dim == 2:
                if point[0] > self.node[0] and point[1] > self.node[1]:
                    ret,dep = self.leafs[0].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] < self.node[1]:
                    ret,dep = self.leafs[1].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] > self.node[1]:
                    ret,dep = self.leafs[2].exists(point,tolerance,depth + 1)
                else:
                    ret,dep = self.leafs[3].exists(point,tolerance,depth + 1)
            else:
                if point[0] > self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[0].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret,dep = self.leafs[1].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[2].exists(point,tolerance,depth + 1)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] < self.node[2]:
                    ret,dep = self.leafs[3].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[4].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret,dep = self.leafs[5].exists(point,tolerance,depth + 1)
                elif point[0] < self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret,dep = self.leafs[6].exists(point,tolerance,depth + 1)
                else:
                    ret,dep = self.leafs[7].exists(point,tolerance,depth + 1)
            if ret == False and dep - depth < 2:
                if self.dim == 2:
                    ret0,dep0 = self.leafs[0].exists(point,tolerance,depth + 1)
                    ret1,dep1 = self.leafs[1].exists(point,tolerance,depth + 1)
                    ret2,dep2 = self.leafs[2].exists(point,tolerance,depth + 1)
                    ret3,dep3 = self.leafs[3].exists(point,tolerance,depth + 1)
                    if ret0 == True:
                        dep = dep0
                        ret = True
                    elif ret1 == True:
                        dep = dep1
                        ret = True
                    elif ret2 == True:
                        dep = dep2
                        ret = True
                    elif ret3 == True:
                        dep = dep3
                        ret = True
                else:
                    ret0,dep0 = self.leafs[0].exists(point,tolerance,depth + 1)
                    ret1,dep1 = self.leafs[1].exists(point,tolerance,depth + 1)
                    ret2,dep2 = self.leafs[2].exists(point,tolerance,depth + 1)
                    ret3,dep3 = self.leafs[3].exists(point,tolerance,depth + 1)
                    ret4,dep4 = self.leafs[4].exists(point,tolerance,depth + 1)
                    ret5,dep5 = self.leafs[5].exists(point,tolerance,depth + 1)
                    ret6,dep6 = self.leafs[6].exists(point,tolerance,depth + 1)
                    ret7,dep7 = self.leafs[7].exists(point,tolerance,depth + 1)
                    if ret0 == True:
                        dep = dep0
                        ret = True
                    elif ret1 == True:
                        dep = dep1
                        ret = True
                    elif ret2 == True:
                        dep = dep2
                        ret = True
                    elif ret3 == True:
                        dep = dep3
                        ret = True
                    elif ret4 == True:
                        dep = dep4
                        ret = True
                    elif ret5 == True:
                        dep = dep5
                        ret = True
                    elif ret6 == True:
                        dep = dep6
                        ret = True
                    elif ret7 == True:
                        dep = dep7
                        ret = True
        else:
            #Search here for it.
            i = 0
            cdis = 0
            cpoint = []
            while i < len(self.skelepts):
                #Gets closest point in container to the search
                if i == 0:
                    cpoint = self.skelepts[i].getPoint()
                    cdis = getDistance(point, cpoint)
                else:
                    tpoint = self.skelepts[i].getPoint()
                    tdis = getDistance(point, tpoint)
                    if tdis < cdis:
                        cpoint = tpoint
                        cdis = tdis
                if cdis < tolerance:
                    self.skelepts[i].addConnection()
                    return True, depth
                i += 1
            return False, depth
        return ret,dep

    def getConnections(self, point : list,*,getpoint = False):
        if self.state:
            #Go Deeper
            if self.dim == 2:
                if point[0] > self.node[0] and point[1] > self.node[1]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] < self.node[1]:
                    ret = self.leafs[1].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] > self.node[1]:
                    ret = self.leafs[2].getConnections(point,getpoint=getpoint)
                else:
                    ret = self.leafs[3].getConnections(point,getpoint=getpoint)
            else:
                if point[0] > self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[0].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret = self.leafs[1].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[2].getConnections(point,getpoint=getpoint)
                elif point[0] > self.node[0] and point[1] < self.node[1] and point[2] < self.node[2]:
                    ret = self.leafs[3].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[4].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] > self.node[1] and point[2] < self.node[2]:
                    ret = self.leafs[5].getConnections(point,getpoint=getpoint)
                elif point[0] < self.node[0] and point[1] < self.node[1] and point[2] > self.node[2]:
                    ret = self.leafs[6].getConnections(point,getpoint=getpoint)
                else:
                    ret = self.leafs[7].getConnections(point,getpoint=getpoint)
        else:
            #Search here for it.
            i = 0
            j = 0
            cdis = 0
            cpoint = []
            while i < len(self.skelepts):
                #Gets closest point in container to the search
                if i == 0:
                    cpoint = self.skelepts[i].getPoint()
                    cdis = getDistance(point, cpoint)
                else:
                    tpoint = self.skelepts[i].getPoint()
                    tdis = getDistance(point, tpoint)
                    if tdis < cdis:
                        cpoint = tpoint
                        cdis = tdis
                        j = i
                i += 1
            if getpoint:
                return self.skelepts[j]
            else:
                return self.skelepts[j].connections
        return ret



                #ty = []
                #while i < len(self.skelepts):
                #    tx.append(self.skelepts[i].x)
                #    ty.append(self.skelepts[i].y)
                #    plt.plot(self.skelepts[i].x + np.cos(theta) * self.skelepts[i].r,self.skelepts[i].y + np.sin(theta) * self.skelepts[i].r,5,color='red')
                #    i += 1
                #plt.scatter(tx,ty,5,color='green')



    def purge(self,incode : int = 0,indata : list = [],*,threshDistance : float = 0.0001):
        #This method is a bit complicated. We pass up and down a code to determine how to act.
        #incode: 0 ->(default, for itterating down the structure),
        #outcode: -1 -> negative space: 0 -> (Bottom level initial information,hard compare): 1 ->
        #print(self.dep)
        retpoints = []#Return Points
        dping = 0#Depth ping
        score = []#Node of real layer
        rdata = []
        intercepts = []
        #Main Logic
        if incode == 0:
            print('depth:',self.dep)
            self.touchstack = []
            if self.state:
                self.stackid = []#Stackid is the kind of data in each ordered leaf
                self.stack = []#Information passed along with the stack
                lowest = 0
                for leaf in self.leafs:#Itterate Down and save results
                    r,l,c = leaf.purge(threshDistance = threshDistance)
                    if l > lowest:
                        lowest = l
                    if c == -1:#layer has nothing beneath
                        self.stackid.append(0)
                        self.stack.append(0)
                    elif c == 0:
                        if isinstance(r[0],SkelePoint):
                            #One point in this layer
                            self.stackid.append(1)
                            self.stack.append(r[0].getPoint())
                        else:
                            #Approx here
                            self.stackid.append(2)
                            self.stack.append(r[0])
                    else:#c==1 so there is a subdivide below
                        self.stackid.append(3)
                        self.stack.append(0)
                if lowest - self.dep == 1:#Only 1 layer beneath isnt enough to resolve
                    rdata = []#We dont need to pass anything
                    outcode = 1
                else:
                    #Now we are in a bigger scope. First we will check for touching points and intercepts.
                    touching = self.getTouch(threshDistance,self.dep)
                    #if theres enough touching, we will invoke a converge results yet we need to make sure to ignore the
                    outcode = 1
                    if (self.dim == 2 and len(touching) > 6) or (self.dim == 3 and len(touching) > 8):#Potential for complete segments, this adjusts stack to what we want
                        self.purge(1,[self.dep,touching],threshDistance=threshDistance)
            else:
                #The Bottom layer of a given sequence. We will fit a Linear line/surface. We will
                #Save these fits in the layer until we want them destroyed
                n = len(self.skelepts)
                lowest = self.dep
                outcode = 0
                if n > 1:
                    #We need to make sure all the points arent equal
                    mark = []
                    i = 0
                    while i < len(self.skelepts) - 1:
                        j = i + 1
                        while j < len(self.skelepts):
                            pti = self.skelepts[i].getPoint()
                            ptj = self.skelepts[j].getPoint()
                            if getDistance(pti,ptj) < threshDistance / 5:
                                mark.append(i)
                                break
                            j += 1
                        i += 1
                    i = len(mark) - 1
                    while i >= 0:
                        del self.skelepts[mark[i]]
                        i -= 1
                    passes = False
                    n = len(self.skelepts)
                    if n > 1:
                        passes = True
                    if self.dim == 2 and passes:
                        #Fits Line y = ax + b
                        sx = 0
                        sx2 = 0
                        sy = 0
                        sxy = 0
                        for pt in self.skelepts:
                            pt = pt.getPoint()
                            sx += pt[0]
                            sx2 += pt[0]*pt[0]
                            sy += pt[1]
                            sxy += pt[0]*pt[1]
                        self.fit = [(n*sxy-sx*sy)/(n*sx2-sx*sx)]#First is a of form ax
                        self.fit.append((sy-self.fit[0]*sx)/n)#now b for ax + b
                        c0y = self.fit[0]*self.c[0][0]+self.fit[1]#y val taken at x of c0
                        c0x = (self.c[0][1] - self.fit[1]) / self.fit[0]#x val taken at y of c0
                        c1y = self.fit[0]*self.c[1][0]+self.fit[1]
                        c1x = (self.c[1][1] - self.fit[1]) / self.fit[0]
                        if c0y > self.c[1][1] and c0y < self.c[0][1]:
                            intercepts.append([self.c[0][0],c0y])
                        if c1y > self.c[1][1] and c1y < self.c[0][1]:
                            intercepts.append([self.c[1][0],c1y])
                        if c0x > self.c[1][0] and c0x < self.c[0][0]:
                            intercepts.append([c0x,self.c[0][1]])
                        if c1x > self.c[1][0] and c1x < self.c[0][0]:
                            intercepts.append([c1x,self.c[1][1]])
                        if len(intercepts) == 3 or len(intercepts) == 0:
                            print('error on int')
                        #We want to adjust intercepts whenever they shoot past points
                        close = []
                        closept = []
                        i = 0
                        while i < len(intercepts):
                            close.append(getDistance(intercepts[0],self.skelepts[0].getPoint()))
                            closept.append(self.skelepts[0].getPoint())
                            j = 1
                            while j < len(self.skelepts):
                                dist = getDistance(intercepts[i],self.skelepts[j].getPoint())
                                if dist < close[i]:
                                    close[i] = dist
                                    closept[i] = self.skelepts[j].getPoint()
                                j += 1
                            i += 1
                        maxd = max(close)
                        mind = min(close)
                        if maxd > 2 * threshDistance:
                            ind = close.index(maxd)
                            lpoint = closept[ind]
                            intercepts[ind] = [(lpoint[0]+intercepts[ind][0])/2,(lpoint[1]+intercepts[ind][1])/2]
                        self.intercepts = intercepts
                    elif self.dim == 3 and passes:
                        #3D's are alittle bit more tricky, as now we have planes Our Solution is to fit a linear plane
                        #We can then find the equations/ directions it crosses in, and we can represent those intercepts and compare to others
                        sx = 0
                        sx2 = 0
                        sy = 0
                        sxy = 0
                        sy2 = 0
                        sxz = 0
                        syz = 0
                        sz = 0
                        for pt in self.skelepts:
                            pt = pt.getPoint()
                            sx += pt[0]
                            sy += pt[1]
                            sz += pt[2]
                            sx2 += pt[0]*pt[0]
                            sxy += pt[0]*pt[1]
                            sy2 += pt[1]*pt[1]
                            sxz += pt[0]*pt[2]
                            syz += pt[1]*pt[2]
                        #Now we fit, Our result will be z = Ax + By + C
                        a = np.array([[sx2,sxy,sx],[sxy,sy2,sy],[sx,sy,n]])
                        b = np.array([sxz,syz,sz])
                        self.fit = np.linalg.lstsq(a,b)[0]
                        self.fit = self.fit.tolist()
                        #Next We Find the Intercepts Going through side of the bounding box
                        #We hold these intercepts in the form of a line equation
                        #IE y = Ax + B where y and x are the planes consistent coordinates
                        #And we will also mark the end points
                        #The expression will be [point1,point2,[A,B]]#We find the plane we are looking by finding conisistence in point1 and point 2
                        #There will be six tests
                        intercepts = []

                        ##YZ plane Approx at x of c0 and c1
                        ax0 = self.fit[0] * self.c[0][0] + self.fit[2]
                        ax1 = self.fit[0] * self.c[1][0] + self.fit[2]
                        #Gets our C* = ax + c value as will remain const
                        # caluclated z = By + C*
                        fitx = []
                        fitx.append([self.fit[1],ax0,self.c[0][0]])
                        fitx.append([self.fit[1],ax1,self.c[1][0]])
                        #Then we find each sectors intercepts
                        for fit in fitx:
                            tcy0 = (self.c[0][2] - fit[1])/fit[0] #yvalue using c0z
                            tcy1 = (self.c[1][2] - fit[1])/fit[0]
                            tcz0 = fit[0]*self.c[0][1] + fit[1]#zvalue using c0y
                            tcz1 = fit[0]*self.c[1][1] + fit[1]
                            #We are positive nothing falls completely out of bounds now, so it must cross the plane
                            tpt = []
                            if tcy0 < self.c[0][1] and tcy0 > self.c[1][1]:#Crosses one one side
                                #Next we find other point
                                tpt.append([fit[2],tcy0,self.c[0][2]])
                            if tcy1 < self.c[0][1] and tcy1 > self.c[1][1]:#Crosses other
                                tpt.append([fit[2],tcy1,self.c[1][2]])
                            if tcz0 < self.c[0][2] and tcz0 > self.c[1][2]:#Crosses one one side
                                #Next we find other point
                                tpt.append([fit[2],self.c[0][1],tcz0])
                            if tcz1 < self.c[0][2] and tcz1 > self.c[1][2]:#Crosses other
                                tpt.append([fit[2],self.c[1][1],tcz1])
                            if len(tpt) > 2 or len(tpt) == 1:
                                print('error on int')
                            elif len(tpt) != 0:
                                #If we make it here we want to take the points
                                intercepts.append([tpt[0],tpt[1],[fit[0],fit[1]],0])
                        #XZ plane Approx at y of c0 and c1
                        ay0 = self.fit[1] * self.c[0][1] + self.fit[2]
                        ay1 = self.fit[1] * self.c[1][1] + self.fit[2]
                        fity = []
                        fity.append([self.fit[0],ay0,self.c[0][1]])
                        fity.append([self.fit[0],ay1,self.c[1][1]])
                        #Then we find each sectors intercepts
                        for fit in fity:
                            tcx0 = (self.c[0][2] - fit[1])/fit[0] #xvalue using c0z
                            tcx1 = (self.c[1][2] - fit[1])/fit[0]
                            tcz0 = fit[0]*self.c[0][0] + fit[1]#zvalue using c0x
                            tcz1 = fit[0]*self.c[1][0] + fit[1]
                            #We are positive nothing falls completely out of bounds now, so it must cross the plane
                            tpt = []
                            if tcx0 < self.c[0][0] and tcx0 > self.c[1][0]:#Crosses one one side
                                #Next we find other point
                                tpt.append([tcx0,fit[2],self.c[0][2]])
                            if tcx1 < self.c[0][0] and tcx1 > self.c[1][0]:#Crosses other
                                tpt.append([tcx1,fit[2],self.c[1][2]])
                            if tcz0 < self.c[0][2] and tcz0 > self.c[1][2]:#Crosses one one side
                                #Next we find other point
                                tpt.append([self.c[0][1],fit[2],tcz0])
                            if tcz1 < self.c[0][2] and tcz1 > self.c[1][2]:#Crosses other
                                tpt.append([self.c[1][1],fit[2],tcz1])
                            if len(tpt) > 2 or len(tpt) == 1:
                                print('error on int')
                            elif len(tpt) != 0:
                                #If we make it here we want to take the points
                                intercepts.append([tpt[0],tpt[1],[fit[0],fit[1]],1])
                        #XY plane Approx at z of c0 and c1
                        az0 = (self.c[0][2] - self.fit[2]) / self.fit[1]
                        az1 = (self.c[1][2] - self.fit[2]) / self.fit[1]
                        #Gets our C* = (z - c)/b value as will remain const
                        #B* = -a/b
                        # caluclated y = B*x + C*
                        fitz = []
                        fitz.append([-(self.fit[0]/self.fit[1]),az0,self.c[0][2]])
                        fitz.append([-(self.fit[0]/self.fit[1]),az1,self.c[1][2]])
                        #Then we find each sectors intercepts
                        for fit in fitz:
                            tcx0 = (self.c[0][1] - fit[1])/fit[0] #xvalue using c0y
                            tcx1 = (self.c[1][1] - fit[1])/fit[0]
                            tcy0 = fit[0]*self.c[0][0] + fit[1] #yvalue using c0x
                            tcy1 = fit[0]*self.c[1][0] + fit[1]
                            #We are positive nothing falls completely out of bounds now, so it must cross the plane
                            tpt = []
                            if tcx0 < self.c[0][0] and tcx0 > self.c[1][0]:#Crosses one one side
                                #Next we find other point
                                tpt.append([tcx0,self.c[0][1],fit[2]])
                            if tcx1 < self.c[0][0] and tcx1 > self.c[1][0]:#Crosses other
                                tpt.append([tcx1,self.c[1][1],fit[2]])
                            if tcy0 < self.c[0][1] and tcy0 > self.c[1][1]:#Crosses one one side
                                #Next we find other point
                                tpt.append([self.c[0][1],tcy0,fit[2]])
                            if tcy1 < self.c[0][1] and tcy1 > self.c[1][1]:#Crosses other
                                tpt.append([self.c[1][1],tcy1,fit[2]])
                            if len(tpt) > 2 or len(tpt) == 1:
                                print('error on int',len(tpt))
                            elif len(tpt) != 0:
                                #If we make it here we want to take the points
                                intercepts.append([tpt[0],tpt[1],[fit[0],fit[1]],2])
                        self.intercepts = intercepts
                    #now we have a 2D or 3D approx
                    #we will want to build some return data, such as a depth ping, intercepts,
                    if passes:
                        rdata = [self.intercepts]
                        lowest = self.dep
                    else:
                        rdata = [self.skelepts[0]]
                        lowest = self.dep
                elif n == 1:
                    #If not enoguh points
                    rdata = [self.skelepts[0]]#When we cant make any approximation, we only have one point, we can test this with other information
                    lowest = self.dep
                else:
                    rdata = []
                    lowest = 0
                    outcode = -1
            #Return Logic
            if self.dep != 0:
                return rdata,lowest,outcode
            else:
                #Final Step
                i = 0
                retpts = []
                retr = []
                for leaf in self.leafs:
                    #We want to go and collect all the points which exist now.
                    #Invoking 4 to get there
                    rp = leaf.purge(4,threshDistance=threshDistance)
                    for p in rp:#Adds in all the points and decomposes them into what we want
                        retpts.append(p.getPoint())
                        retr.append(p.getRad())
                self.Draw(self.dep)
                return retpts,retr
        elif incode == 4:#4 used for quick add
            retdata = []
            if self.state:
                for leaf in self.leafs:
                    rd = leaf.purge(4,threshDistance=threshDistance)
                    for r in rd:
                        retdata.append(r)
            else:
                if len(self.skelepts) > 0:
                    if isinstance(self.skelepts[0],list):
                        retdata.append(skelepts[0][0])
                    else:
                        for p in self.skelepts:
                            retdata.append(p)
            return retdata
        elif incode == 1:
            if self.dep == indata[0]:#top layer does more
                #Full Touches
                for t in indata[1]:
                    t0 = t[0]
                    t1 = t[1]
                    it0 = (len(t0) == 1)
                    it1 = (len(t1) == 1)
                    if it0 and it1:
                        self.leafs[t0[0]].touchstack.append([-1,t1])
                        if len(self.leafs[t0[0]].touchstack) > 2:
                            self.stackid[t0[0]] = -1
                        self.leafs[t1[0]].touchstack.append([-1,t0])
                        if len(self.leafs[t1[0]].touchstack) > 2:
                            self.stackid[t1[0]] = -1
                    elif it0:
                        self.leafs[t0[0]].touchstack.append([-1,t1])
                        if len(self.leafs[t0[0]].touchstack) > 2:
                            self.stackid[t0[0]] = -1
                        self.leafs[t1[0]].purge(1,[indata[0],[t1[1],[-1,t0]]],threshDistance=threshDistance)
                    elif it1:
                        self.leafs[t1[0]].touchstack.append([-1,t0])
                        if len(self.leafs[t1[0]].touchstack) > 2:
                            self.stackid[t1[0]] = -1
                        self.leafs[t0[0]].purge(1,[indata[0],[t0[1],[-1,t1]]],threshDistance=threshDistance)
                    else:
                        if t0[0] == t1[0]:
                            self.leafs[t0[0]].purge(1,[indata[0],[t0[1],t1[1]],0],threshDistance=threshDistance)
                        else:
                            self.leafs[t0[0]].purge(1,[indata[0],[t0[1],t1[1]]],threshDistance=threshDistance)
                            self.leafs[t1[0]].purge(1,[indata[0],[t1[1],t0[1]]],threshDistance=threshDistance)
                #Now that we have marked, any points that dont fall within threshDistance of the edge of highest scop should be able to be corrected
                self.purge(2,[self.dep],threshDistance=threshDistance)
            else:#Non top level touch adder
                #first of indata[1] is the path we want
                t0 = indata[1][0]
                t1 = indata[1][1]
                it0 = (len(t0) == 1)
                it1 = (len(t1) == 1)
                if len(indata) == 3:#We want to target both t0 and t1 individually
                    if it0 and it1:
                        self.leafs[t0[0]].touchstack.append([-1,t1])
                        if len(self.leafs[t0[0]].touchstack) > 2:
                            self.stackid[t0[0]] = -1
                        self.leafs[t1[0]].touchstack.append([-1,t0])
                        if len(self.leafs[t1[0]].touchstack) > 2:
                            self.stackid[t1[0]] = -1
                    elif it0:
                        self.leafs[t0[0]].touchstack.append([-1,t1])
                        if len(self.leafs[t0[0]].touchstack) > 2:
                            self.stackid[t0[0]] = -1
                        self.leafs[t1[0]].purge(1,[indata[0],[t1[1],[-1,t0]]],threshDistance=threshDistance)
                    elif it1:
                        self.leafs[t1[0]].touchstack.append([-1,t0])
                        if len(self.leafs[t1[0]].touchstack) > 2:
                            self.stackid[t1[0]] = -1
                        self.leafs[t0[0]].purge(1,[indata[0],[t0[1],[-1,t1]]],threshDistance=threshDistance)
                    else:
                        if t0[0] == t1[0]:
                            self.leafs[t0[0]].purge(1,[indata[0],[t0[1],t1[1]],0],threshDistance=threshDistance)
                        else:
                            self.leafs[t0[0]].purge(1,[indata[0],[t0[1],t1[1]]],threshDistance=threshDistance)
                            self.leafs[t1[0]].purge(1,[indata[0],[t1[1],t0[1]]],threshDistance=threshDistance)
                else:#Here were targeting t0 with t1
                    if it0:
                        self.leafs[t0[0]].touchstack.append([-1,t1])
                    else:
                        self.leafs[t0[0]].purge(1,[indata[0],[t0[1],[-1,t1]]],threshDistance=threshDistance)
        if incode == 2:
            if self.dep == indata[0]:
                indata.append(self.c)
            i = 0
            for sid in self.stackid:
                if sid == 1:
                    lts = len(self.leafs[i].touchstack)
                    if self.dim == 2:
                        if lts == 0:
                            if indata[0] == 0 or (self.dep) - indata[0] > 2:#ensure scope
                                self.stack[i] = 0
                                self.stackid[i] = 0
                                self.leafs[i].skelepts = []
                elif sid == 2:
                    lts = len(self.leafs[i].touchstack)
                    if self.dim == 2:
                        if lts == 0:
                            if indata[0] == 0 or (self.dep) - indata[0] > 2:#ensure scope
                                self.stack[i] = 0
                                self.stackid[i] = 0
                                self.leafs[i].skelepts = []
                elif sid == 3:#Go Further
                    self.leafs[i].purge(2,indata,threshDistance=threshDistance)
                i += 1

####UNUSED########################################################################################################################################################
        elif incode == 3:
            #Now we have our data, so lets go to the scope we want
            if len(indata[1]) > 1:
                senddata = indata.copy()
                i = 0
                while i < len(senddata[3]):
                    if senddata[3][i][0] == senddata[1][0]:
                        senddata[3][i] = senddata[3][i][1]
                    else:
                        senddata[3][i] = [-1,senddata[3][i]]
                    i += 1
                senddata[1] = senddata[1][1]
                self.leafs[indata[1][0]].purge(3,senddata,threshDistance=threshDistance)
            else:
                #We have arrived at the scope we want
                #First we will sort though the indata, and make sure none of the points we are getting exist in this scope
                i = 0
                while i < len(indata[2]):
                    if self.stackid[indata[1][0]] == 1:
                        if len(self.stack[indata[1][0]]) == 1:
                            if indata[2][i] == self.stack[indata[1][0]][0]:
                                del indata[2][i]
                                del indata[3][i]
                        else:
                            if indata[2][i] == self.stack[indata[1][0]]:
                                del indata[2][i]
                                del indata[3][i]
                    else:
                        for p in self.leafs[indata[1][0]].skelepts:
                            pt = p.getPoint()
                            if pt == indata[2]:
                                del indata[2][i]
                                del indata[3][i]
                                break
                    i += 1
                if self.stackid[indata[1][0]] == 1:
                    i = 0
                    while i < len(indata[3]):
                        go = True
                        for tpath in self.leafs[indata[1][0]].touchstack:
                            if tpath == [-1,indata[3][i]]:
                                go = False
                                break
                        if go:
                            #If not already touching, then
                            if len(self.stack[indata[1][0]]) == 1:
                                if getDistance(self.stack[indata[1][0]][0],indata[2][i]) < threshDistance:
                                    self.leafs[indata[1][0]].touchstack.append([-1,indata[3][i]])
                            else:
                                if getDistance(self.stack[indata[1][0]],indata[2][i]) < threshDistance:
                                    self.leafs[indata[1][0]].touchstack.append([-1,indata[3][i]])
                            if self.stackid[indata[1][0]] != -1 and len(self.leafs[indata[1][0]].touchstack) > 1:
                                self.stackid[indata[1][0]] = -1
                        i += 1
                    if indata[0] == 0 or self.dep - indata[0] > 1:
                        #We have permission to hard delete
                        if len(self.leafs[indata[1][0]].touchstack) == 0:
                            self.stackid[indata[1][0]] = 0
                            self.stack[indata[1][0]] = 0
                            self.leafs[indata[1][0]].skelepts = []
                        else:
                            print()
                            print('one left')
                            print(self.c)
                            print(self.stack)
                            print(self.stackid)
                            print(self.leafs[indata[1][0]].touchstack)
                            print()
                if self.stackid[indata[1][0]] == 2:
                    i = 0
                    while i < len(indata[3]):
                        go = True
                        for tpath in self.leafs[indata[1][0]].touchstack:
                            if tpath == [-1,indata[3][i]]:
                                go = False
                                break
                        if go:
                           #We want to go through each point and compare to the current touch attempt, If the point isnt inline then :(
                            for inter in self.stack[indata[1][0]]:
                                if getDistance(indata[2][i],inter) < threshDistance:
                                    self.leafs[indata[1][0]].touchstack.append([-1,indata[3][i]])
                        i += 1
                    cpts = []
                    if indata[0] == 0 or self.dep - indata[0] > 1:
                        tp = self.leafs[indata[1][0]].skelepts
                        approx = self.leafs[indata[1][0]].fit
                        #We have permission to hard delete
                        if len(self.leafs[indata[1][0]].touchstack) < 2:
                            #If we have one or no touches we want to take a closer look, because we should have enough theoretically here
                            i = 0
                            if len(self.leafs[indata[1][0]].touchstack) == 1:
                                tpt = []
                                for p in tp:
                                    tpt.append(p.getPoint())
                                        #We have **maybe 50% right maybe none of it and we need to rotate

                                    i += 1
                                #We will try rotation to check points.
                        else:#We already have touches, so test points for linearity
                            tp = self.leafs[indata[1][0]].skelepts
                            i = 0
                            tpt = []
                            for p in tp:
                                #We assume approx is 100% correct here as we have enough touches, take away any major deviations
                                tpt.append(p.getPoint())
                                if self.dim == 2:
                                    if getDistance(tpt[len(tpt)-1],[tpt[len(tpt)-1][0],approx[0]*tpt[len(tpt)-1][0]+approx[1]]) > threshDistance:
                                        tpt = tpt[:-1]
                                        del self.leafs[indata[1][0]].skelepts[i]
                                        if len(self.leafs[indata[1][0]].skelepts) == 1:
                                            self.stackid[indata[1][0]] = 1
                                            self.stack[indata[1][0]] = self.leafs[indata[1][0]].skelepts[0].getPoint()
                                        if len(self.leafs[indata[1][0]].skelepts) == 0:
                                            self.stackid[indata[1][0]] = 0
                                            self.stack[indata[1][0]] = 0
                                            self.leafs[indata[1][0]].skelepts = []
                                i += 1
    def deepDive(self,indepth : int ,target : list, bounds : list,*,threshDistance:float):
        #Deep dive will go through the tree and find all the touching node path points, ie closest to the container we want
        #We will input a target node path, and the bounds of said path, and it will output potential connections for said node to connect to
        #Potential connections are ranked
        results = []
        paths = []
        if self.state:#If Divided
            goto = []
            if bounds[0][0] >= self.node[0] - threshDistance:
                #Right side
                if bounds[0][1] >= self.node[1] - threshDistance:#TR
                    if self.dim == 2:
                        goto.append(0)
                    else:
                        if bounds[0][2] >= self.node[2] - threshDistance:
                            print('temp')
                        if bounds[1][2] <= self.node[2] + threshDistance:
                            print('temp')
                if bounds[1][1] <= self.node[1] + threshDistance:#BR
                    if self.dim == 2:
                        goto.append(1)
                    else:
                        if bounds[0][2] >= self.node[2] - threshDistance:
                            print('temp')
                        if bounds[1][2] <= self.node[2] + threshDistance:
                            print('temp')
            if bounds[1][0] <= self.node[0] + threshDistance:
                if bounds[0][1] >= self.node[1] - threshDistance:#TL
                    if self.dim == 2:
                        goto.append(2)
                    else:
                        if bounds[0][2] >= self.node[2] - threshDistance:
                            print('temp')
                        if bounds[1][2] <= self.node[2] + threshDistance:
                            print('temp')
                if bounds[1][1] <= self.node[1] + threshDistance:#BL
                    if self.dim == 2:
                        goto.append(3)
                    else:
                        if bounds[0][2] >= self.node[2] - threshDistance:
                            print('temp')
                        if bounds[1][2] <= self.node[2] + threshDistance:
                            print('temp')
            #We now have the nodes that cound potentially hold something
            for loc in goto:
                sid = self.stackid[loc]
                if sid == 3:
                    res,pth = self.leafs[loc].deepDive(indepth,target,bounds,threshDistance=threshDistance)
                    if isinstance(res,list):
                        for r in res:
                            results.append(r)
                        for p in pth:
                            paths.append([loc,p])
                    else:
                        print('error on res')
                elif sid == -1 or (sid != 0 and len(self.leafs[loc].touchstack) > 0):
                    #Will search if converged or has a touch, prevents considering anything from
                    #Unknown Sectors
                    if isinstance(self.stack[loc][0],float):
                        #If one point, then the stack is the point, and thus we will consider it as a true point
                        results.append(self.stack[loc])
                        paths.append([loc])
                    else:
                        if self.dim == 2:
                            best = [self.stack[loc][0][0],self.stack[loc][0][1]]
                        else:
                            best = [self.stack[loc][0][0],self.stack[loc][0][1],self.stack[loc][0][2]]
                        bestdis = getDistance(best,target)
                        for st in self.stack[loc]:
                            if getDistance(st,target) < bestdis:
                                best = st
                                bestdis = getDistance(st,target)
                        results.append(best)
                        paths.append([loc])
                else:
                    if sid != 0 and len(self.leafs[loc].touchstack) != 0:
                        print()
                        print('type',sid)
                        print(len(self.leafs[loc].touchstack))
                        print(self.stack[loc])
                        print()
            if self.dep != indepth:
                return results, paths
            else:
                return results,paths
####UNUSED############################################################################################################################################################



    def getTouch(self,threshDistance,indepth):
        #get touch is implied that there has already been a stack build
        touchid = []
        extrema = [[],[]]#Extrema holds [[points/intercepts],[trace id's]]
        if max(self.stackid) < 3:
            #There are no more splits below. So we just single test each
            i = 0
            while i < len(self.stackid) - 1:
                j = i + 1
                if self.stack[i] != 0 and self.stack[i] != [] and isinstance(self.stack[i][0],float):
                    if self.dim == 2:
                        self.stack[i] = [[self.stack[i][0],self.stack[i][1]]]
                    else:
                        self.stack[i] = [[self.stack[i][0],self.stack[i][1],self.stack[i][2]]]
                while j < len(self.stackid):
                    if self.stack[j] != 0 and self.stack[j] != []  and isinstance(self.stack[j][0],float):
                        if self.dim == 2:
                            self.stack[j] = [[self.stack[j][0],self.stack[j][1]]]
                        else:
                            self.stack[j] = [[self.stack[j][0],self.stack[j][1],self.stack[j][2]]]
                    idi = self.stackid[i]
                    idj = self.stackid[j]
                    if idi != 0 and idj != 0:
                        #prevents empty compare
                        if idi == 2 and idj == 2:
                            #2 approx
                            for int1 in self.stack[i]:
                                for int2 in self.stack[j]:
                                    if self.dim == 2:
                                        if getDistance(int1,int2) < threshDistance:
                                            touchid.append([i,j])
                                    else:
                                        #we grab which planet it is on, 0 -> yz plane
                                        p1 = int1[2]
                                        p2 = int2[2]
                                        if int1[3] == int2[3] and np.abs(p1[0]-p2[0]) < threshDistance and np.abs(p1[1]-p2[1]) < threshDistance:
                                            touchid.append([i,j])
                                        #We also add if a point is close to it :)
                                        elif getDistance(int1[0],int2[0]) < threshDistance or getDistance(int1[0],int2[1]) < threshDistance:
                                            touchid.append([i,j])
                                        elif getDistance(int1[1],int2[0]) < threshDistance or getDistance(int1[1],int2[1]) < threshDistance:
                                            touchid.append([i,j])
                        elif idi == 1 and idj == 1:
                            #1 comp
                            if getDistance(self.stack[i][0],self.stack[j][0]) < threshDistance:
                                touchid.append([i,j])
                        else:#makes sure both arent already converged, no point to checking if they are
                            #3 comp
                            if (idi == -1 and idj != -1) or (idj == -1 and idi != -1):
                                #We are dealing with a converged sequence
                                if idi == -1:
                                    typestack = len(self.stack[i])
                                    if typestack == 1:
                                        if idj == 1:
                                            if getDistance(self.stack[i][0],self.stack[j][0]) < threshDistance:
                                                tcase = True
                                                for touches in self.leafs[i].touchstack:
                                                    if touches == [-1,[j]]:
                                                        tcase = False
                                                        break
                                                if tcase:
                                                    touchid.append([i,j])
                                        else:
                                            for int2 in self.stack[j]:
                                                if self.dim == 2:
                                                    if getDistance(self.stack[i][0],int2) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[i].touchstack:
                                                            if touches == [-1,[j]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                                else:
                                                    if getDistance(self.stack[i][0],int2[0]) < threshDistance or getDistance(self.stack[i][0],int2[1]) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[i].touchstack:
                                                            if touches == [-1,[j]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                    else:
                                        for int1 in self.stack[i]:
                                            if idj == 1:
                                                if self.dim == 2:
                                                    if getDistance(self.stack[i][0],int1) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[i].touchstack:
                                                            if touches == [-1,[j]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                                else:
                                                    if getDistance(self.stack[j][0],int1[0]) < threshDistance or getDistance(self.stack[j][0],int1[1]) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[i].touchstack:
                                                            if touches == [-1,[j]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                            else:
                                                for int2 in self.stack[j]:
                                                    if self.dim == 2:
                                                        if getDistance(int1,int2) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[i].touchstack:
                                                                if touches == [-1,[j]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                                    else:
                                                        p1 = int1[2]
                                                        p2 = int2[2]
                                                        if int1[3] == int2[3] and np.abs(p1[0]-p2[0]) < threshDistance and np.abs(p1[1]-p2[1]) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[i].touchstack:
                                                                if touches == [-1,[j]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                                        elif getDistance(int1[0],int2[0]) < threshDistance or getDistance(int1[0],int2[1]) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[i].touchstack:
                                                                if touches == [-1,[j]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                                        elif getDistance(int1[1],int2[0]) < threshDistance or getDistance(int1[1],int2[1]) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[i].touchstack:
                                                                if touches == [-1,[j]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                else:
                                    typestack = len(self.stack[j])
                                    if typestack == 1:
                                        if idi == 1:
                                            if getDistance(self.stack[i][0],self.stack[j][0]) < threshDistance:
                                                tcase = True
                                                for touches in self.leafs[j].touchstack:
                                                    if touches == [-1,[i]]:
                                                        tcase = False
                                                        break
                                                if tcase:
                                                    touchid.append([i,j])
                                        else:
                                            for int2 in self.stack[i]:
                                                if self.dim == 2:
                                                    if getDistance(self.stack[j][0],int2) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[j].touchstack:
                                                            if touches == [-1,[i]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                                else:
                                                    if getDistance(self.stack[j][0],int2[0]) < threshDistance or getDistance(self.stack[j][0],int2[1]) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[j].touchstack:
                                                            if touches == [-1,[i]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                    else:
                                        for int1 in self.stack[j]:
                                            if idi == 1:
                                                if self.dim == 2:
                                                    if getDistance(int1,self.stack[i][0]) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[j].touchstack:
                                                            if touches == [-1,[i]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])
                                                else:
                                                    if getDistance(self.stack[i][0],int1[0]) < threshDistance or getDistance(self.stack[i][0],int1[1]) < threshDistance:
                                                        tcase = True
                                                        for touches in self.leafs[j].touchstack:
                                                            if touches == [-1,[i]]:
                                                                tcase = False
                                                                break
                                                        if tcase:
                                                            touchid.append([i,j])

                                            else:
                                                for int2 in self.stack[i]:
                                                    if self.dim == 2:
                                                        if getDistance(int1,int2) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[j].touchstack:
                                                                if touches == [-1,[i]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                                    else:
                                                        p1 = int1[2]
                                                        p2 = int2[2]
                                                        if int1[3] == int2[3] and np.abs(p1[0]-p2[0]) < threshDistance and np.abs(p1[1]-p2[1]) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[j].touchstack:
                                                                if touches == [-1,[i]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                                        elif getDistance(int1[0],int2[0]) < threshDistance or getDistance(int1[0],int2[1]) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[j].touchstack:
                                                                if touches == [-1,[i]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                                                        elif getDistance(int1[1],int2[0]) < threshDistance or getDistance(int1[1],int2[1]) < threshDistance:
                                                            tcase = True
                                                            for touches in self.leafs[j].touchstack:
                                                                if touches == [-1,[i]]:
                                                                    tcase = False
                                                                    break
                                                            if tcase:
                                                                touchid.append([i,j])
                            elif idi == 1:
                                #idi is the single
                                for int2 in self.stack[j]:
                                    if self.dim == 2:
                                        if getDistance(self.stack[i][0],int2) < threshDistance:
                                            touchid.append([i,j])
                                    else:
                                        if getDistance(self.stack[i][0],int2[0]) < threshDistance or getDistance(self.stack[i][0],int2[1]) < threshDistance:
                                            touchid.append([i,j])
                            elif idj == 1:
                                #idj is single
                                for int1 in self.stack[i]:
                                    if self.dim == 2:
                                        if getDistance(self.stack[j][0],int1) < threshDistance:
                                            touchid.append([i,j])
                                    else:
                                        if getDistance(self.stack[j][0],int1[0]) < threshDistance or getDistance(self.stack[j][0],int1[1]) < threshDistance:
                                            touchid.append([i,j])
                    j += 1
                i += 1
            #Now we check for extrema
            i = 0
            while i < len(self.stackid):
                idi = self.stackid[i]
                if idi == 1:
                    for corner in self.c:
                        if self.dim == 2:
                            if abs(self.stack[i][0][0]-corner[0]) < threshDistance or abs(self.stack[i][0][1]-corner[1]) < threshDistance:
                                extrema[0].append(self.stack[i])
                                extrema[1].append(i)
                                break
                        else:
                            if abs(self.stack[i][0][0]-corner[0]) < threshDistance or abs(self.stack[i][0][1]-corner[1]) < threshDistance or abs(self.stack[i][0][2]-corner[2]):
                                extrema[0].append(self.stack[i])
                                extrema[1].append(i)
                                break

                elif idi == 2:
                    for inter in self.stack[i]:
                        for corner in self.c:
                            if self.dim == 2:
                                if abs(inter[0]-corner[0]) < threshDistance or abs(inter[1]-corner[1]) < threshDistance:
                                    extrema[0].append(inter)
                                    extrema[1].append(i)
                                    break
                            else:
                                if abs(inter[0][0]-corner[0]) < threshDistance or abs(inter[0][1]-corner[1]) < threshDistance or abs(inter[0][2]-corner[2]):
                                    extrema[0].append(self.stack[i][0])
                                    extrema[1].append(i)
                                if abs(inter[1][0]-corner[0]) < threshDistance or abs(inter[1][1]-corner[1]) < threshDistance or abs(inter[1][2]-corner[2]):
                                    extrema[0].append(self.stack[i][1])
                                    extrema[1].append(i)

                i += 1
            return touchid,extrema
        else:
            #We have a point which is split, we will get all touches from those first, then compare with our non touches
            i = 0
            while i < len(self.stackid):
                if self.stackid[i] == 3:
                    tid,e = self.leafs[i].getTouch(threshDistance,indepth)
                    j = 0
                    while j < len(tid):#adds touching path
                        if isinstance(tid[j][0],int):
                            touchid.append([[i,[tid[j][0]]]])
                        else:
                            touchid.append([[i,tid[j][0]]])
                        if isinstance(tid[j][1],int):
                            touchid[len(touchid) - 1].append([i,[tid[j][1]]])
                        else:
                            touchid[len(touchid) - 1].append([i,tid[j][1]])
                        j += 1
                    j = 0
                    while j < len(e[0]):#adds in extrema types
                        if isinstance(e[1][j],int):
                            extrema[1].append([i,[e[1][j]]])
                        else:
                            extrema[1].append([i,e[1][j]])
                        if isinstance(e[0][j],list):
                            extrema[0].append(e[0][j])
                        else:
                            print('whoops')
                        j += 1
                elif self.stackid[i] == 1:
                    if isinstance(self.stack[i][0],list):
                        extrema[0].append(self.stack[i])
                    else:
                        extrema[0].append(self.stack[i])
                    extrema[1].append(i)
                elif self.stackid[i] == 2:
                    for inter in self.stack[i]:
                        if self.dim == 2:
                            extrema[0].append(inter)
                            extrema[1].append(i)
                        else:
                            extrema[0].append(inter[0])
                            extrema[1].append(i)
                            extrema[0].append(inter[1])
                            extrema[1].append(i)
                elif self.stackid[i] == -1:
                    if len(self.stack[i]) > 0:
                        if isinstance(self.stack[i][0],list):
                            for inter in self.stack[i]:
                                if self.dim == 2:
                                    if isinstance(inter,float):
                                        print('error')
                                    extrema[0].append(inter)
                                    extrema[1].append(i)
                                else:
                                    if isinstance(inter[0],float):
                                        print('error')
                                    if isinstance(inter[1],float):
                                        print('error')
                                    extrema[0].append(inter[0])
                                    extrema[1].append(i)
                                    extrema[0].append(inter[1])
                                    extrema[1].append(i)
                i += 1
            i = 0
            e0 = extrema[0]
            e1 = extrema[1]
            while i < len(e0):
                if not isinstance(e0[i][0],float):
                    e0[i] = e0[i][0]
                i += 1
            i = 0
            while i < len(e0) - 1:
                j = i + 1
                while j < len(e0):
                    if getDistance(e0[i],e0[j]) < threshDistance:
                        if isinstance(e1[i],int):
                            e1[i] = [e1[i]]
                        if isinstance(e1[j],int):
                            e1[j] = [e1[j]]
                        touchid.append([e1[i],e1[j]])
                    j += 1
                i += 1
            if self.dep != indepth:
                te = [[],[]]
                i = 0
                while i < len(e0):
                    for corner in self.c:
                        if self.dim == 2:
                            if abs(e0[i][0] - corner[0]) < threshDistance or abs(e0[i][1] - corner[1]) < threshDistance:
                                te[0].append(e0[i])
                                te[1].append(e1[i])
                        else:
                            if abs(e0[i][0] - corner[0]) < threshDistance or abs(e0[i][1] - corner[1]) < threshDistance or abs(e0[i][2] - corner[2]) < threshDistance:
                                te[0].append(e0[i])
                                te[1].append(e1[i])
                    i += 1
                extrema = te
                return touchid,extrema
            else:
                i = 0
                j = 0
                tempid = []
                while i < len(touchid) - 1:
                    j = i + 1
                    case = False
                    while j < len(touchid):
                        if touchid[i] == touchid[j]:
                            case = True
                            break
                        j += 1
                    if case == False:
                        tempid.append(touchid[i])
                    i += 1
                touchid = tempid
                return touchid
    def Draw(self,indep : int = 0,name : str = r'quadplot.png',*,index : int = 0):
        #This is the class for creating a visual of the quad tree structure
        #First Collect Data
        rdata = []
        if self.state:
            rdata.append([self.dep,self.c[0],self.c[1]])
            #only care about upper nodes, as they are the divided ones
            i = 0
            for leaf in self.leafs:
                if self.stackid[i] != 0:
                    results = leaf.Draw(index = i)
                    for res in results:
                        rdata.append(res)
                i += 1
        else:
            if len(self.skelepts) > 0:
                rdata.append([self.dep,self.c[0],self.c[1]])
                rpt = []
                for pt in self.skelepts:
                    rpt.append(pt)
                rdata[0].append(rpt)
                rdata[0].append(index)
                if len(self.skelepts) > 1:
                    #there is an approx made, so we will grab it
                    rdata[0].append(self.intercepts)

        #Next we either draw or return
        if self.dep == indep:
            if self.dim == 2:
                plt.clf()
                plt.rcParams['figure.dpi'] = 300
                nodes = []
                deps = []
                for data in rdata:
                    #Itterates through everything
                    #plt.plot([data[1][0],data[1][0],data[2][0],data[2][0],data[1][0]],[data[1][1],data[2][1],data[2][1],data[1][1],data[1][1]],c='black')
                    if len(data) > 3:
                        #Drawing bottom layer box with points
                        tpts = []
                        for pt in data[3]:
                            if data[4] == 0:
                                plt.scatter(pt.x,pt.y,s=5,c='red')
                            elif data[4] == 1:
                                plt.scatter(pt.x,pt.y,s=5,c='blue')
                            elif data[4] == 2:
                                plt.scatter(pt.x,pt.y,s=5,c='green')
                            else:
                                plt.scatter(pt.x,pt.y,s=5,c='purple')
                    #if len(data) > 5:
                        #plt.plot([data[5][0][0],data[5][1][0]],[data[5][0][1],data[5][1][1]],c='green')


            save = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + r'/Plot/' + name
            plt.savefig(save)
            if self.dim == 2:
                plt.clf()
                plt.rcParams['figure.dpi'] = 300
                nodes = []
                deps = []
                for data in rdata:
                    #Itterates through everything
                    plt.plot([data[1][0],data[1][0],data[2][0],data[2][0],data[1][0]],[data[1][1],data[2][1],data[2][1],data[1][1],data[1][1]],c='black')
                    if len(data) > 3:
                        #Drawing bottom layer box with points
                        tpts = []
                        for pt in data[3]:
                            if data[4] == 0:
                                plt.scatter(pt.x,pt.y,s=5,c='red')
                            elif data[4] == 1:
                                plt.scatter(pt.x,pt.y,s=5,c='blue')
                            elif data[4] == 2:
                                plt.scatter(pt.x,pt.y,s=5,c='green')
                            else:
                                plt.scatter(pt.x,pt.y,s=5,c='purple')
                    if len(data) > 5:
                        plt.plot([data[5][0][0],data[5][1][0]],[data[5][0][1],data[5][1][1]],c='green')


            save = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + r'/Plot/' + 'box' + name
            plt.savefig(save)
        else:
            return rdata

class SkelePoint:
#This is a class which has a point which holds x,y,z and r

    def __init__(self,point : list,*, rad : float = 0.0,connections : int = 0):
        if len(point) != 0:
            self.x = point[0]
            self.y = point[1]
            self.r = rad
            self.connections = connections
            if len(point) == 3:
                self.z = point[2]
                self.dimensions = 3
            else:
                self.dimensions = 2
            self.ordered = False
    def getPoint(self):
        if self.dimensions == 2:
            return [self.x,self.y]
        else:
            return [self.x,self.y,self.z]

    def getRad(self):
        return self.r

    def addConnection(self):
        self.connections += 1

    def getAxis(self,dim : int):
        if dim == 0:
            return self.x
        elif dim == 1:
            return self.y
        elif self.dimensions == 3:
            if dim == 2:
                return self.z
    def locEx(self,point):
        if point == 0:
            return False
        elif isinstance(point,list):
            if len(point) == 0:
                return False
            pt = self.getPoint()
            return pt == point
        else:
            i = 0
            while i < self.dimensions:
                if self.getAxis(i) != point.getAxis(i):
                    return False
                i += 1
        return True




















