import os
from random  import randint
# from sys import float_repr_style
# import matplotlib
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import numpy as np
import csv
# import scipy
import pandas as pd
import time
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool,ThreadPool

from DataStructures import kdTree,SplitTree
from Skeletize import checkRepeat,getRadius,getDistance,normalize, getAngle, getPoint
from mpl_toolkits import mplot3d
from itertools import cycle
cycol = cycle('bgrcmk')

# @profile
def skeletize(points : list,norms : list,threshDistance : float,tree : kdTree,animate : bool = False,cpuid : int = -1,tag : int = -1,*,cpuavail : int = 1):
    print('process{} started'.format(cpuid))
    #Skeletize takes in 
    #FROM INPUT
    #key is the specific index of tpoints and tnorms, allows for
    #parallel capabilities in splitting apart skeleton tasks 
    #Also allows the class to have an array being filled by different tags
    #FROM CLASS
    #points, the given plain list of points [x,y] for 2D case
    #norms, a list of not yet normalized normal points [n_x,n_y] here for 2D case
    #then returns 2 things
    # finPoints = [[x1,y1],...] of skeleton points
    # finR = [r1,...] of the radius of each skeleton point
    ts = time.time()
    # if len(self.SkelePoints) != key + 1:
    #     self.SkelePoints.append([])
    #     self.SkeleRad.append([])
    #     print('Skeletizing #{}...'.format(key))
    

    ##INITAL SETTING UP METHOD
    #removes the connection between input points and output ponts
    #When not doing this kd-tree skews points for some reason?
    avgt = 0
    numt = 0
    dim = len(points[0])
    guessr = threshDistance * len(points) * 10
    ##START OF SOLVE
    index = 0
    lenptso = len(str(len(points)))
    avgstep = 0
    prnd = []
    pts = []
    nrms = []
    #For parallelizing, we must give it everything it needs. 
    SkelePoints = []
    SkeleRad = []
    if animate:
        acp = []
        atp = []
        arad = []
    i = 0
    while i < len(points):
        pts.append(points[i])
        nrms.append(norms[i])
        i += 1
    i = 0
    
    for point in pts:
        stt = time.time()
        #finding inital temp radius
        norm = nrms[index]
        tempr = []
        i = 0
        centerp = []
        testp = []
        #print(index,cpuid) 
        case = False
        # print('Tag:{},Id:{}'.format(tag,cpuid),index,'/',len(pts) - 1,'{}%'.format((index / (len(pts) - 1)) * 100))
        #Main loop for each points solve
        while not case:
            if i == 0:
                #Inital
                tempr.append(guessr)
                if tempr[0] < 0.015:
                    print(index,'flag')
                if len(points[0]) == 2:
                    centerp.append([float(point[0]-norm[0]*tempr[0]),float(point[1]-norm[1]*tempr[0])])
                else:
                    centerp.append([float(point[0]-norm[0]*tempr[0]),float(point[1]-norm[1]*tempr[0]),float(point[2]-norm[2]*tempr[0])])
                testp.append(point)
            else:   
                #Refinement of skeleton point
                # testp.append(tree.getNearR(centerp[len(centerp) - 1],point,cpuavail=cpuavail))
                tpool = ThreadPool(cpuavail)
                inputdat = []
                inputdat.append([])
                inputdat[0].append(centerp[len(centerp) - 1])
                inputdat[0].append(point)
                inputdat[0].append(False)
                results = tpool.map(tree.getNearR,inputdat)
                testp.append(results[0])
                tpool.close()
                if testp[len(testp) - 1] == point:
                    print()
                    print('error',index,i)
                    print('centers',centerp)
                    print('testp',testp)
                    print('rads',tempr)
                    print('Point',point,'Norm',norm)
                    print()
                    
                tempr.append(np.round(getRadius(point,testp[i],norm),6))
                if dim == 2:
                    centerp.append([float(point[0]-norm[0]*tempr[i]),float(point[1]-norm[1]*tempr[i])])
                else:
                    centerp.append([float(point[0]-norm[0]*tempr[i]),float(point[1]-norm[1]*tempr[i]),float(point[2]-norm[2]*tempr[i])])
            leng = len(tempr) - 1
                            #Capture animation data
            if animate:
                if i == 0:
                    acp.append([])
                    atp.append([])
                    arad.append([])
                    acp[index].append(centerp[0])
                    atp[index].append(testp[0])
                    arad[index].append(tempr[0])
                else:
                    acp[index].append(centerp[leng])
                    atp[index].append(testp[leng])
                    arad[index].append(tempr[leng])
                    
            #Checking for completeion
            
            #Convergence check
            dist = getDistance(point,testp[leng])
            if i > 1 and np.abs(tempr[leng] - tempr[leng - 1]) < threshDistance:
                if tempr[leng] < (threshDistance) or dist < (tempr[leng]) + threshDistance:
                    SkelePoints.append(centerp[leng - 1])
                    SkeleRad.append(tempr[leng - 1])
                    #Show backstep in animation
                    if animate:
                        acp[index].append(centerp[leng - 1])
                        atp[index].append(testp[leng - 1])
                        arad[index].append(tempr[leng - 1])
                else:
                    # print('norm')
                    SkelePoints.append(centerp[leng])
                    SkeleRad.append(tempr[leng])
                
                case = True 
            
            #Overshooting  
            elif i > 1 and tempr[leng] < (threshDistance):
                SkelePoints.append(centerp[leng - 1])
                SkeleRad.append(tempr[leng - 1])
                #Show backstep in animation
                if animate:
                    acp[index].append(centerp[leng - 1])
                    atp[index].append(testp[leng - 1])
                    arad[index].append(tempr[leng - 1])
                case = True
            elif i > 0 and dist  < tempr[leng] + threshDistance:
                #Checks if the point is closer than the cross point if it falls here, alittle expensive but should fix errors
                crossdis = 0
                tpoint = []
                tnorm = []
                j = 0
                while j < len(point):
                    tnorm.append(norm[j] * -1)
                    tpoint.append(point[j] + tnorm[j] * threshDistance)
                    j += 1
                # crossp = tree.getVectorR(tpoint,tnorm,1,scan = (np.pi / 8),cpuavail=cpuavail)
                tpool = ThreadPool(cpuavail)
                inputdat = []
                inputdat.append([])
                inputdat[0].append(tpoint)
                inputdat[0].append(tnorm)
                inputdat[0].append(1)
                inputdat[0].append(False)
                inputdat[0].append((np.pi / 8))
                results = tpool.map(tree.getVectorR,inputdat)
                crossp = results[0]
                tpool.close
                if len(crossp) > 0:
                    crossp = crossp[0] 
                    crossdis = getDistance(point,crossp)
                print(crossp)
                if getDistance(point,centerp[leng]) < crossdis:
                    SkelePoints.append(centerp[leng - 1])
                    SkeleRad.append(tempr[leng - 1])
                    #Show backstep in animation
                    if animate:
                        acp[index].append(centerp[leng - 1])
                        atp[index].append(testp[leng - 1])
                        arad[index].append(tempr[leng - 1])
                    case = True
            
            
            #Repeat check
            elif i > 3:
                #Really only comes up with perfect shapes,
                #but always a possibility to happen
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
                   print('Repeat')
                   print(point,testp,tempr,centerp)
                   SkeleRad.append(sml)
                   SkelePoints.append(centerp[n])
                   case = True 
            i += 1
        avgt += (time.time() - stt)
        avgstep += len(tempr)
        numt += 1
        if index % 10 == 0:
            tat = avgt / numt
            sat = avgstep / numt
            print('CPUID:{:02d} TAG:{:02d} || T-Time:{:04.2f}h:{:04.2f}m:{:04.2f}s || A-Time:{:04.2f}m:{:04.2f}s || {}/{} {:04.2f}%-Done avgstep:{:02d}'.format(cpuid,tag,avgt // 3600, (avgt % 3600) // 60,(avgt % 3600) % 60,tat // 60,tat % 60,str(index + 1).zfill(lenptso),len(points), ((index + 1) / (len(points))) * 100,int(np.ceil(sat)))) 
        index += 1
    te = time.time()
    tt = te - ts
    print('Skeleton took {} minuites and {} seconds'.format((tt) // 60,(tt) % 60))
    if animate:
        return SkelePoints,SkeleRad,acp,atp,arad
    else:
        return SkelePoints,SkeleRad

class SkeleNet:
    #In simpleTerms Skelenet is an easy to use skeletonization processer, 
    #It can intake a location of a data file, or even the straight points
    #Then itll process then and output different figures for different things
    #Can also Produce different shapes and stuff
    rnd = 0
###INITALIZERS
    def __init__(self, points,*,norms = []):
        #Solve variables, certain parts of these can be thrown away later on depending id heap is heavy
        self.IntPoints = []
        self.NormPoints = []
        self.MasterTag = []
        self.tagKey = []
        self.tpoints = []
        self.tnorms = []
        self.threshDistance = []
        self.orderData = []
        #Final Variables (for now depending on how we later edit this information)
        self.SkelePoints = []
        self.SkeleRad = []
        self.tagged = False
        
        
        #Multiprocessing ideas
        self.cpuavail = min(mp.cpu_count() - 4,28) #Will Always allow 2 Cores to remain unused
        if self.cpuavail == 0:
            self.cpuavail = 1
        print('We have {} CPU\'s Available'.format(self.cpuavail))
        #Determining type of points given
        if isinstance(points,str):    
            with open(points,'r') as csvfile:
                data = csv.reader(csvfile, delimiter = ' ')
                for row in data:
                    
                    size = len(row)
                    if str(row[0]) == 'x':#if title
                        a = 1    
                    elif size == 4:#2D w/ no tag
                        self.IntPoints.append([float(row[0]),float(row[1])])
                        self.NormPoints.append([float(row[2]),float(row[3])])
                        self.MasterTag.append(0)
                    elif size == 5:#2D w/ tag
                        self.IntPoints.append([float(row[0]),float(row[1])])
                        self.NormPoints.append([float(row[2]),float(row[3])]) 
                        self.MasterTag.append(int(row[4]) - 1)
                    elif size == 6:#3D w/ no tag
                        self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                        self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
                        self.MasterTag.append(0)
                    elif size == 7:#3D w/ tag
                        self.IntPoints.append([float(row[0]),float(row[1]),float(row[2])])
                        self.NormPoints.append([float(row[3]),float(row[4]),float(row[5])])
                        self.MasterTag.append(int(row[6]) - 1)
            csvfile.close()
            
        elif isinstance(points,list):
            for point in points:
                self.IntPoints.append(point)
            if isinstance(norms,list):
                for norm in norms:
                    self.NormPoints.append(norm)
        

        if len(self.IntPoints[0]) == 3:
            self.dim = 3
        else:
            self.dim = 2
        
        #Gets normals if needed then normalizes the entire vectors
        if not(len(self.NormPoints) > 1):
            self.getNorms()#Caution, normals only acurate to locals and perfect shapes and such, use for Smooth Points

        temp  = normalize(self.NormPoints)
        self.NormPoints = []
        self.NormPoints = temp
        
        self.__tag()

###MAIN FUNCTIONS
    

    def solve(self,animate : bool = False):
        st = time.time()
        #Solves for all taggs individually
        #will be paralled in the future
        if animate:
            self.animate = True
            self.acp = []
            self.atp = []
            self.arad = []
        else:
            self.animate = False
        #First we want to make all the nessicary trees and distances
        i = 0
        self.tree = []
        while i < len(self.tpoints):
            #Tree Made
            pts = []
            j = 0
            while j < len(self.tpoints[i]):
                pts.append(self.tpoints[i][j])
                j += 1
            self.tree.append(kdTree(pts))
            
            #Thresh distance is defined
            tot = 0
            tpt = []
            j = 0
            while j < min(20,len(self.tpoints[i])):
                tpt = self.tpoints[i][randint(0, len(self.tpoints[i]) - 1)]
                tot += getDistance(tpt,self.tree[i].getNearR([tpt,[],False]))
                j += 1
            self.threshDistance.append(tot / min(20,len(self.tpoints[i]))) 
            print('thresh',self.threshDistance)  
            i += 1
            
        
        # self.divpts = []
        # self.divnrms = []
        # strt = []
        # stp = []
        # i = 0
        # while i < len(self.tpoints):
        #     j = 0
        #     self.divpts.append([])
        #     self.divnrms.append([])
        #     strt.append([])
        #     stp.append([])
        #     while j < self.cpuavail:
        #         self.divpts[i].append([])
        #         self.divnrms[i].append([])
        #         q = int(np.floor((len(self.tpoints[i]) - 1) * j / self.cpuavail))
        #         strt[i].append(int(np.floor((len(self.tpoints[i]) - 1) * j / self.cpuavail)))
        #         stp[i].append(int(np.floor((len(self.tpoints[i]) - 1) * (j+1) /self.cpuavail)))
        #         while q < int(np.floor((len(self.tpoints[i]) - 1) * (j+1) /self.cpuavail)):
        #             self.divpts[i][j].append(self.tpoints[i][q])
        #             self.divnrms[i][j].append(self.tnorms[i][q])
        #             q += 1
        #         j += 1  
        #     i += 1
        i = 0
        while i < len(self.tpoints):    
            # self.pool = ProcessingPool(nodes=self.cpuavail)
            # setthresh = []
            # temptree = []
            # ani = []
            # cpuid = []
            # cputag = []
            # j = 0
            # while j < len(self.divpts[i]):
            #     setthresh.append(self.threshDistance[i])
            #     temptree.append(self.tree[i])
            #     ani.append(self.animate)
            #     cpuid.append(j)
            #     cputag.append(i)
            #     j += 1
            print('booting processes...')
            centp,rad = skeletize(self.tpoints[i], self.tnorms[i], self.threshDistance[i], self.tree[i],cpuavail = self.cpuavail)
            # results = self.pool.map(skeletize,self.divpts[i],self.divnrms[i],setthresh,temptree,ani,cpuid,cputag)
            print('done')
            # for data in results:
            #     t0 = data[0]
            #     t1 = data[1]
            #     if self.animate:
            #         t2 = data[2]
            #         t3 = data[3]
            #         t4 = data[4]
            #     if len(self.SkelePoints) == i:
            #         self.SkelePoints.append([])
            #         self.SkeleRad.append([])
            #     j = 0
            #     while j < len(t0):
            #         self.SkelePoints[i].append(t0[j])
            #         self.SkeleRad[i].append(t1[j])
            #         j += 1
            #     if self.animate:
            #         if len(self.acp) == i:
            #             self.acp.append([])
            #         if len(self.atp) == i:
            #             self.atp.append([])
            #         if len(self.arad) == i:
            #             self.arad.append([])
            #         j = 0
            #         while j < len(t2):
            #             self.acp[i].append(t2[j])
            #             j += 1
            #         j = 0
            #         while j < len(t3):
            #             self.atp[i].append(t3[j])
            #             j += 1
            #         j = 0
            #         while j < len(t4):
            #             self.arad[i].append(t4[j])
            #             j += 1
            i += 1
            self.pool.close()
        # self.order()
        et = time.time()
        tt = (et - st)
        print('Total Solve took: {} Minuites {} Seconds'.format(tt // 60, tt % 60))
        
        
    def order(self):
        #This function will go through all of the points 
        st = time.time()
        t = 0
        self.Otrees = []
        self.Strees = []
        self.delpoints = []
        self.orderpoints = []
        #Created the needed Trees to fidn the direction 
        while t < len(self.SkelePoints):
            self.Otrees.append(kdTree(self.SkelePoints[t], self.dim,rads=self.SkeleRad[t]))
            ret,extra = self.orderR(t)
            i = 0
            extra[1].append([])
            ends = []
            while i < len(ret):
                #There should be two matching points close to eachother in the return, could be n number of lists tho, but there would be an equal point.
                ends.append(ret[i][0][0])
                ends.append(ret[i][len(ret[i]) - 1][0])
                i += 1
            i = 0
            closeends = []
            while i < len(ends):
                j = i + 1
                while j < len(ends):
                    if getDistance(ends[i], ends[j]) < 5 * self.threshDistance[t]:
                        closeends.append(ends[i])
                        closeends.append(ends[j])
                    j += 1
                i += 1
            i = 0
            while i < len(ret):
                if i == 0:
                    q = 0
                    while q < len(closeends):
                        if closeends[q] == ret[i][0][0]:
                            ret[i].reverse()
                            break
                        q += 1
                    q = 0
                    while q < len(ret[i]):
                        extra[1][len(extra[1]) - 1].append(ret[i][q])
                        q += 1
                else:
                    q = 0
                    while q < len(closeends):
                        if closeends[q] == ret[i][len(ret[i]) - 1][0]:
                            ret[i].reverse()
                            break
                        q += 1
                    q = 0
                    while q < len(ret[i]):
                        extra[1][len(extra[1]) - 1].append(ret[i][q])
                        q += 1
                i += 1
            self.orderpoints = extra[1]
            t += 1
        #This method is designed to search, order, and reduce the skeleton points into simple informat
        #Using a Depth-First Search It will recreate surfaces  
        
        

        et = time.time()
        tt = et - st
        print('Ordering took {} minuites and {} seconds'.format((tt) // 60,(tt) % 60))
    def orderR(self,key : int,depth : int = 0,point : list = [],lastNode : list = []):
        
        #First grabs a random point from the given Skeleton data to take as the Original Point
        Local = []#local describes all points within a 10*threshdistance range
        Localr = []#locals radii
        
        if depth == 0:
            point =  self.SkelePoints[key][randint(0,len(self.SkelePoints[key]) - 1)]
            
            
            
        Local,Localr = self.Otrees[key].getInR(point,self.threshDistance[key],1,getRads = True)
        leng = len(Local)
        vast = self.Otrees[key].getInR(point,self.threshDistance[key] * 10,1)
        lengv =  len(vast)
        
        #Check for closeness
        avgx = 0
        avgr = 0
        avgy = 0
        avgz = 0
        i = 0
        while i < leng:
            avgx += Local[i][0]
            avgy += Local[i][1]
            avgr += Localr[i]
            if self.dim == 3:
                avgz += Local[i][2]
            i += 1
        avgx = avgx/leng
        avgy = avgy/leng
        avgr = avgr/leng
        if self.dim == 3:
            avgz = avgz/leng
            nodep = [avgx,avgy,avgz]
        else:
            nodep = [avgx,avgy]
  
        #Boot up the stack if unmade
        if len(self.Strees) == key:
            #Find Maxbounds
            if self.dim == 2:
                minx,miny,maxx,maxy=self.tpoints[key][0][0],self.tpoints[key][0][1],self.tpoints[key][0][0],self.tpoints[key][0][1]
                for pt in self.tpoints[key]:
                    maxx = pt[0] if pt[0] > maxx else maxx
                    maxy = pt[1] if pt[1] > maxy else maxy
                    minx = pt[0] if pt[0] < minx else minx
                    miny = pt[1] if pt[1] < miny else miny
                center = [minx + (maxx-minx) / 2,miny + (maxy-miny) / 2]
                width = max(np.abs(maxx-minx),np.abs(maxy-miny))
                self.Strees.append(SplitTree([nodep],center, width / 2, inrad = [avgr]))
            else:
                minx,miny,minz,maxx,maxy,maxz=self.tpoints[key][0][0],self.tpoints[key][0][1],self.tpoints[key][0][2],self.tpoints[key][0][0],self.tpoints[key][0][1],self.tpoints[key][0][2]
                for pt in self.tpoints[key]:
                    maxx = pt[0] if pt[0] > maxx else maxx
                    maxy = pt[1] if pt[1] > maxy else maxy
                    maxz = pt[2] if pt[2] > maxz else maxz
                    minx = pt[0] if pt[0] < minx else minx
                    miny = pt[1] if pt[1] < miny else miny
                    minz = pt[2] if pt[2] < minz else minz
                center = [minx + (maxx-minx) / 2,miny + (maxy-miny) / 2,minz + (maxz-minz) / 2]
                width = max(np.abs(maxx-minx),np.abs(maxy-miny),np.abs(maxz-minz))
                self.Strees.append(SplitTree([nodep],[0,0,0], width / 2, inrad = [avgr]))
        else:
            self.Strees[key].addpoints([nodep],rads = [avgr])
        
        #The Next step is getting directional information and determining branches nearby 
        #First creating realitive direction vectors
        i = 0
        n = 8#n determines how many sub divisions there are
        dirv = []
        while i < n:
            theta = i * (2 * np.pi / n)
            if self.dim == 2:
                dirv.append([np.round(np.cos(theta),6),np.round(np.sin(theta),6)])
            else:
                j = 0
                while j < n:
                    #isnt great, needs fixes for 3D
                    phi = j * (2 * np.pi / n)
                    dirv.append([np.round(np.sin(theta)*np.cos(phi),6),np.round(np.sin(theta)*np.sin(phi),6),np.round(np.cos(theta),6)])
                    j += 1
            i += 1
            
        #Deletes repeat vectors
        if self.dim == 3:
            tdirv = []
            #Needs to thin off extra vectors
            i = 0
            tleng = len(dirv)
            while i < tleng:
                j = i + 1
                case = False
                if i == tleng - 1:
                    case = True
                else:
                    while j < tleng:
                        if dirv[i] == dirv[j]:
                            case = False
                            break
                        else:
                            case = True
                        j += 1
                if case:
                    tdirv.append(dirv[i])
                i += 1
            dirv = tdirv
        tempdir = []
        lasttag = 10000000
        q = 0
        #gets points in directions
        for vec in dirv:
            lpvec = []
            vecpoint = []
            #Gets 20 closeest points in each direction
            i = 0
            while i < len(vec):
                if not(depth == 0):
                    lpvec.append(lastNode[i] - point[i])
                if vec[i] == -0.0:
                    vec[i] = 0.0
                if not(depth == 0):
                    vecpoint.append(point[i] + vec[i])
                i += 1
            tempdir.append(self.Otrees[key].getVectorR(point,vec,10,getRads=True))
            if not(depth == 0):
                if getAngle(lpvec,vec,getDistance(point,lastNode),getDistance(point,vecpoint)) < np.pi / 8:
                    lasttag = q
            q += 1
        checkedtags = []
        emptytags = []#Empty tags are nodes that should be ignored alltogether as they have no points
        combtags = []#Comb tags is a collection of connected tags
        isotags = []#Iso tags give us a good idea of if there is a branch
        case = True
        
        #Sorts collections into connected pieces, isolated pieces, and empty pieces.
        #isolated collections must have some sort of branch, as no other options could exist at given step
        current = 0
        while case:
            checkedtags.append(current)
            #Checks for empty directions and marks them, grabs next point
            if len(tempdir[current][0]) == 0:
                emptytags.append(current)
            #If node is not empty, should determine if there is connected pieces or the direction is isolated
            else:
                
                #Collect nearby tags. if they are empy tag them as such now, if they have points, attach them to nearby's
                neartags = []
               
                i = 0
                while i < len(dirv):
                    if np.abs(getAngle(dirv[current],dirv[i],getDistance(point,getPoint(point,dirv[current])),getDistance(point, getPoint(point,dirv[i]))) - np.pi / 4) < self.threshDistance[key]:
                        neartags.append(i)
                    i += 1
                
                #Sorts through the nearby ones, adding it to 
                skipped = []
                empties = 0
                skips = 0
                for tag in neartags:
                    #Prevents reChecking Points, If skipping will add to 
                    j = 0
                    skip = True
                    while j < len(checkedtags):
                        if tag == checkedtags[j]:
                            skipped.append(tag)
                            if len(tempdir[tag][0]) == 0:
                                empties += 1
                            skip = False
                            skips += 1
                            break
                        j += 1
                    if skip:
                        checkedtags.append(tag)
                        if len(tempdir[tag][0]) == 0:
                            emptytags.append(tag)
                            empties += 1
                        else:
                            if len(combtags) == 0:
                                combtags.append([])
                                combtags[0].append(tag)
                                combtags[0].append(current)
                            else:
                                j = 0
                                found = False
                                while j < len(combtags):
                                    k = 0
                                    if not(len(skipped) == 0):
                                        while k < len(combtags[j]):
                                            p = 0
                                            while p < len(skipped):
                                                if combtags[j][k] == current or combtags[j][k] == skipped[p]:
                                                    combtags[j].append(tag)
                                                    found = True
                                                    break
                                                p += 1
                                            if found:
                                                break
                                            k += 1
                                    else:
                                        while k < len(combtags[j]):
                                            if combtags[j][k] == current:
                                                combtags[j].append(tag)
                                                found = True
                                                break
                                            k += 1
                                    if found:
                                        break
                                    j += 1
                                if not(found):
                                    combtags.append([])
                                    combtags[len(combtags) - 1].append(current)
                                    combtags[len(combtags) - 1].append(tag)
                if len(skipped) > 0:
                    i = 0
                    found = False
                    while i < len(skipped):
                        j = 0
                        while j < len(combtags):
                            k = 0
                            while k < len(combtags[j]):
                                if combtags[j][k] == skipped[i]:
                                    found = True
                                    combtags[j].append(current)
                                    break
                                k += 1
                            if found:
                                break
                            j += 1
                        if found:
                            break
                        i += 1
                    if len(combtags) > 1:
                        #Needs to check if needs to merge nodes
                        mergenodes = []
                        k = 0
                        while k < len(skipped):
                            j = 0
                            while j < len(combtags):
                                testmerge = False
                                q = 0
                                while q < len(combtags[j]):
                                    if combtags[j][q] == skipped[k]:
                                        mergenodes.append(j)
                                        testmerge = True
                                        break
                                    q += 1
                                if testmerge:
                                    break
                                j += 1
                            k += 1 
                        if len(mergenodes) > 1:
                            tempnew = []
                            k = 0
                            while k < len(mergenodes):
                                q = 0
                                while q < len(combtags[mergenodes[k]]):
                                    tempnew.append(combtags[mergenodes[k]][q])
                                    q += 1
                                k += 1
                            tempcomb = []
                            k = 0
                            while k < len(combtags):
                                q = 0
                                keep = True
                                while q < len(mergenodes):
                                    if k == mergenodes[q]:
                                        keep = False
                                        break
                                    q += 1
                                if keep:
                                    tempcomb.append(combtags[k])
                                k += 1
                            tempcomb.append(tempnew)
                            combtags = tempcomb
                if empties == len(neartags):
                    isotags.append(current)
            if len(checkedtags) == len(tempdir):
                #Only triggers when all points have been checked
                case = False
            else:
                i = 0
                while i < len(tempdir):
                    j = 0
                    det = False
                    while j < len(checkedtags):
                        if i == checkedtags[j]:
                            i += 1
                            det = False
                            break
                        else:
                            det = True
                        j += 1
                    if i == len(checkedtags) or det:
                        current = i 
                        break
        
        
        #Now we have generalized vector collections, Empties will be ignored, combos will be considered together
        #Iso's will be treated as simple branches and stepped out upon
        branches = 0
        newNodes = []
        lengiso = len(isotags)
        if lengiso > 0:
            #See if the next points are about near the average next step
            i = 0
            while i < lengiso:
                if not(isotags[i] == lasttag):
                    isopts,isorads = tempdir[isotags[i]]
                    q = 0
                    mindis = 0
                    minpoint = []
                    while q < len(isopts):
                        tpoint = isopts[q]
                        
                        tdis = getDistance(point,tpoint)
                        if q == 0:
                            mindis = tdis
                            minpoint = tpoint
                        else:
                            if tdis < mindis:
                                minpoint = tpoint
                                mindis = tdis
                        q += 1

                    #Iso points must always continue. even if its one point and far away. it will tag for destruction
                    #So we dont care if it is close enough. We will step regardless, errors will be located later.
                    #If the distance is less than 4 * thresh, we step along that vector and get nearest. if its more
                    #than that, we will just go directly to that point
                    if mindis < 2 * self.threshDistance[key]:
                        travelvec = []
                        temppoint = []
                        q = 0
                        while q < len(minpoint):
                            travelvec.append(minpoint[q] - point[q])
                            q += 1
                        [travelvec] = normalize([travelvec])
                        q = 0
                        while q < len(minpoint):
                            temppoint.append(point[q] + travelvec[q] * 2 * self.threshDistance[key])
                            q += 1
                        newNodes.append(self.Otrees[key].getNearR(temppoint,point))
                    else:
                        newNodes.append(minpoint)
                else:
                    branches += 1
                i += 1

        lengcomb = len(combtags)
        if lengcomb > 0:
            #Determines which leafs get close enough to the point to be branches, as thats all we care about 
            i = 0
            while i < lengcomb:
                j = 0
                while j < len(combtags[i]):
                    if not(combtags[i][j] == lasttag):
                        combpts,combrads = tempdir[combtags[i][j]]
                        q = 0
                        mindis = 0
                        minpoint = []
                        while q < len(combpts):
                            tpoint = combpts[q]
                            tdis = getDistance(point,tpoint)
                            if q == 0:
                                mindis = tdis
                                minpoint = tpoint
                            else:
                                if tdis < mindis:
                                    minpoint = tpoint
                                    mindis = tdis
                            q += 1
                        if mindis < 2 * self.threshDistance[key]:
                            #We want to capture a branch here
                            travelvec = []
                            temppoint = []
                            q = 0
                            while q < len(minpoint):
                                travelvec.append(minpoint[q] - point[q])
                                q += 1
                            [travelvec] = normalize([travelvec])
                            q = 0
                            while q < len(minpoint):
                                temppoint.append(point[q] + travelvec[q] * 2 * self.threshDistance[key])
                                q += 1
                            newNodes.append(self.Otrees[key].getNearR(temppoint,point))
                        elif len(combpts) < 3 and getDistance(point, minpoint) < 20 * self.threshDistance[key]:
                            newNodes.append(minpoint)
                    else:
                        branches += 1
                    j += 1
                i += 1
                
        #Here we gather information from further down the line, if leng = 0 and there are no iso tags, then the program mostlikely didnt step
        output = []
        extra = []
        visited = []
        if not((lengv == 0 and len(isotags) == 0)):
            i = 0
            while i < len(newNodes):
                exists,dep = self.Strees[key].exists(newNodes[i],self.threshDistance[key])
                if not(exists):
                    #this node hasnt been visited yet(verified with stack), should take a step in that direction
                    visited.append(newNodes[i])
                    out, ex = self.orderR(key,depth + 1,newNodes[i],point)
                    for o in out:
                        output.append(o)
                    q = len(extra)
                    while q < 2:
                        extra.append([])
                        q += 1
                    for e in ex[1]:
                        extra[0].append(e)
                    for e in ex[1]:
                        extra[1].append(e)
                    dotest = True
                    j = 0
                    while j < len(extra[0]):
                        if extra[0][j] == newNodes[i]:
                            dotest = False
                            break
                        j += 1
                    if dotest:
                        q = 0
                        while q < len(output):
                            #Tags all 'LastPoints' which may be too big of a differing radius 
                            if np.abs(output[q][len(output[q]) - 1][1] - avgr) > self.threshDistance[key] * 5:
                                extra[0].append(output[q][len(output[q]) - 1][0])
                            q += 1
                    branches += 1#This counts all the connected branches at this point. branches can also be connected in a
                                 #later state if needed         
                i += 1
        else:
            print('LENGTH=0')
            branches += 1
            closestp = self.Otrees[key].getNearR(point,[])
            exists,dep = self.Strees[key].exists(closestp,self.threshDistance[key])
            if not(exists):
                visited.append(closestp)
                out, ex = self.orderR(key,depth + 1,closestp,point)   
                for o in out:
                    output.append(o)
                q = len(extra)
                while q < 2:
                    extra.append([])
                    q += 1
                for e in ex[1]:
                    extra[0].append(e)
                for e in ex[1]:
                    extra[1].append(e)
                j = 0
                dotest = True
                while j < len(extra[0]):
                    if extra[0][j] == closestp:
                        dotest = False
                        break
                    j += 1
                if dotest:
                    q = 0
                    while q < len(output):
                        if np.abs(output[q][len(output[q]) - 1][1] - avgr) > self.threshDistance[key] * 5:
                            extra[0].append(output[q][len(output[q]) - 1][0])
                        q += 1
                            
                            
        #Output layerd in [node point, radius],.. until a 3 section point is found.
        #Extra is stored in [[flagged points],[found branches]], branches => [[branch, 0],[branch,00],[branch, 01]..ect]
        #For further clarity a branch is then [[StartPoint(w rad?)],[EndPoint(w rad?)],[Location Coef],[Rad Coef]]
        #However now, a branch is just a collection of the [nodepoint,radius]
        # print()
        # print('visited',visited)
        if branches == 0:
            print(depth,'Error, No Where To Go')
            #No Branches is a bad thing anywhere; this shouldnt happen
        elif branches == 1:
            # print(depth,point,'stop')
            #This is a complete Stop point. it has gone the deepest it can go. 
            #Adds a flag if not close to anything
            i = len(extra)
            while i < 2:
                extra.append([])
                i += 1
            if lengv == 1:
                extra[0].append(nodep)
            return [[[nodep,avgr]]], extra
        elif branches > 2:
            #3 or more branches means some sort of node branching
            # print(depth,'split')
            i = len(extra)
            while i < 2:
                extra.append([])
                i += 1
            if lengv == 1:
                extra[0].append(nodep)
            
            #First gathers nodes its gone to, stores a list of all and ones we want to visit
            j = 0
            comparenodes = []
            startnodes = []
            while j < len(visited):
                tnodec = self.Strees[key].getConnections(visited[j],getpoint=True).getPoint()
                startnodes.append(tnodec)
                i = 0
                flagged = False
                while i < len(extra[0]):
                    if extra[0][i] == tnodec:
                        flagged = True
                        self.delpoints.append(tnodec)
                        break
                    i += 1
                if not(flagged):
                    comparenodes.append(tnodec)
                j += 1
            # print('comparenodes',comparenodes)
            #Now we search through the outputs and add in the branches we want
            i = 0
            while i < len(output):
                j = 0
                while j < len(comparenodes):
                    if output[i][len(output[i]) - 1][0] == comparenodes[j]:
                        #This is a part we want to save
                        extra[1].append(output[i])
                        break
                    j += 1
                i += 1
            output = []
            output.append([[nodep,avgr]])
            extra[0] = []
            # print('extra',extra[1])
            return output, extra
        else:
            # print(depth,'continue')
            #2 branches, should continue onwards
            #adds flag if not close to anything
            i = len(extra)
            while i < 2:
                extra.append([])
                i += 1
            if lengv == 1:
                extra[0].append(nodep)
            output[0].append([nodep,avgr])
            return output, extra
    
   
                
                
                
    
        
###MISC FUCNTIONS FOR SKELENET
    def __tag(self):
        i = 0
        #Organizing Points means going through and seperating everything by tag
        while i < len(self.IntPoints):
            if i > 0:
                found = False
                j = 0
                for tag in self.tagKey:
                    size = len(self.tagKey)
                    if self.MasterTag[i] == tag:
                        found = True
                        self.tpoints[j].append(self.IntPoints[i])
                        self.tnorms[j].append(self.NormPoints[i])
                    j += 1
                if not(found):
                    self.tagKey.append(self.MasterTag[i])
                    self.tpoints.append([])
                    self.tnorms.append([])
                    self.tpoints[size].append(self.IntPoints[i])
                    self.tnorms[size].append(self.NormPoints[i])  
            else:  
                self.tagKey.append(self.MasterTag[i])
                self.tpoints.append([])
                self.tnorms.append([])
                self.tpoints[0].append(self.IntPoints[i])
                self.tnorms[0].append(self.NormPoints[i])
            i += 1
            
    def __getNorms(self):
        tree = kdTree(self.IntPoints,self.dim)
        for point in self.IntPoints:
            if self.dim == 2:#2D Norms
                close1 = tree.getNearR(point,[])
                close2 = tree.getNearR(point,close1)
                if point == self.IntPoints[0]:
                    print(close1,close2)
                normA = [close1[1] - point[1],close1[0] - point[0]]
                normB = [close2[1] - point[1],close2[0] - point[0]]
                normP = [-1 * (normA[0] + normB[0]) / 2,-1 * (normA[1] + normB[1]) / 2]
                if point == self.IntPoints[0]:
                    print(normP)
                self.NormPoints.append(normP)
            else:#3D Norms
                return    

    ####ImageProcessing
    def plot(self,mode : list = [],*,norm = True,tag = 'None',start : int = 0,stop : int = 9999):
        sst = time.time()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        theta = np.linspace(0,2*np.pi,100)
        index = 0
        tt = 0
        i = 0
        xmax = -10
        xmin = 10
        ymax = -10
        ymin = 10
        while i < len(self.IntPoints):
            if self.IntPoints[i][0] > xmax:
                xmax = self.IntPoints[i][0]
            if self.IntPoints[i][0] < xmin:
                xmin = self.IntPoints[i][0]
            if self.IntPoints[i][1] > ymax:
                ymax = self.IntPoints[i][1]
            if self.IntPoints[i][1] < ymin:
                ymin = self.IntPoints[i][1]
            i += 1
        xmax = np.round(xmax,1)
        xmin = np.round(xmin,1)
        ymax = np.round(ymax,1)
        ymin = np.round(ymin,1)
        xdis = xmax - xmin
        ydis = ymax - ymin
        if xdis > ydis:
            ycent = (ymax + ymin) / 2
            ymin = ycent - xdis / 2
            ymax = ycent + xdis / 2
        else:
            xcent = (xmax + xmin) / 2
            xmin = xcent - ydis / 2
            xmax = xcent + ydis / 2
        print([xmax,xmin],[ymax,ymin])
        while index < len(mode):
            print("Plotting {}".format(mode[index]))
            st = time.time()
            #Mode 0 -> output to degbug of normals of each point
            if mode[index] == 0:
                plt.clf()
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                tx = []
                ty = []
                i = 0
                while i < len(self.IntPoints):
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                i = 0
                while i < len(self.IntPoints):
                    print(i,'/',len(self.IntPoints) - 1)
                    plt.scatter(tx,ty)
                    plt.scatter(self.IntPoints[i][0],self.IntPoints[i][1]) 
                    if norm == False:
                        plt.plot([self.IntPoints[i][0] + self.NormPoints[i][0] * 1000,self.IntPoints[i][0] + self.NormPoints[i][0] * - 1000],[self.IntPoints[i][1] + self.NormPoints[i][1] * 1000,self.IntPoints[i][1] + self.NormPoints[i][1] * -1000])
                    else:
                        plt.plot([self.IntPoints[i][0] + self.NormPoints[i][0] * 1000,self.IntPoints[i][0]],[self.IntPoints[i][1] + self.NormPoints[i][1] * 1000,self.IntPoints[i][1]])
                    save = os.getcwd() + "\Debug\Debug{:04d}.png".format(i)
                    plt.savefig(save)
                    plt.clf()
                    plt.xlim(xmin,xmax)
                    plt.ylim(ymin,ymax)
                    i += 1
                
            #Mode 1 is for outputting final points for every tag
            elif mode[index] == 1:
                plt.clf()
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                if self.dim == 2:
                    i = 0
                    tx = []
                    ty = []
                    while i < len(self.IntPoints):    
                        tx.append(self.IntPoints[i][0])
                        ty.append(self.IntPoints[i][1])
                        i += 1
                    plt.scatter(tx,ty,5,color='blue')
                    i = 0
                    #theta = np.linspace(0,2*np.pi)
                    while i < len(self.SkelePoints):
                        j = 0
                        tx = []
                        ty = []
                        tr = []
                        while j < len(self.SkelePoints[i]):
                            tx.append(self.SkelePoints[i][j][0])
                            ty.append(self.SkelePoints[i][j][1])
                            tr.append(self.SkeleRad[i][j])
                            j += 1
                        i += 1
                        plt.scatter(tx,ty,5,color='orange')
                        plt.savefig('OutputNoRad.png')
                        j = 0
                        while j < len(tx):
                            plt.plot(tx[j] + np.cos(theta) * tr[j],ty[j] + np.sin(theta) * tr[j],5,color='green')
                            j += 1
                else:
                    ax = plt.axes(projection='3d')
                    i = 0
                    tx = []
                    ty = []
                    tz = []
                    while i < len(self.IntPoints):
                        tx.append(self.IntPoints[i][0])
                        ty.append(self.IntPoints[i][1])
                        tz.append(self.IntPoints[i][2])
                        i += 1
                    ax.scatter3D(tx,ty,tz,5,color='red')
                    i = 0
                    while i < len(self.SkelePoints):
                        j = 0
                        tx = []
                        ty = []
                        tz = []
                        tr = []
                        while j < len(self.SkelePoints[i]):
                            tx.append(self.SkelePonts[i][j][0])
                            ty.append(self.SkelePonts[i][j][1])
                            tz.append(self.SkelePonts[i][j][2])
                            tr.append(self.SkeleRad[i][j])
                            j += 1
                        ax.scatter3D(tx,ty,tz,5,c=tr,cmap='winter')
                        i += 1
                plt.savefig('Output.png')
            #Mode2 is for Animating the process of solving
            elif mode[index] == 2:
                svnum = 0
                plt.clf()
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                path = os.getcwd()
                tag = 0
                i = 0
                case = True
                path = path + "/AnimationData/"
                while case:
                    tpath = path + f'{i:04d}' + '/'
                    if not(os.path.isdir(tpath)):
                        case = False
                        path = tpath
                        os.mkdir(tpath)
                    i += 1
                i = 0
                tx = []
                ty = []
                while i < len(self.IntPoints):    
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                sx = []
                sy = []
                while tag < len(self.acp):
                    i = start
                    while i < len(self.acp[tag]):
                        j = 0
                        while j < len(self.acp[tag][i]):
                            plt.clf()
                            plt.xlim(xmin,xmax)
                            plt.ylim(ymin,ymax)
                            print(tag, '/', len(self.acp),' ', i ,'/' , len(self.acp[tag]), ' ', j , '/', len(self.acp[tag][i]))
                            plt.scatter(tx,ty,5,color='green')
                            if len(sx) > 0:
                                plt.scatter(sx,sy,5,color='orange')
                            plt.plot([self.acp[tag][i][j][0],self.tpoints[tag][i][0]],[self.acp[tag][i][j][1],self.tpoints[tag][i][1]])
                            plt.plot([self.atp[tag][i][j][0],self.tpoints[tag][i][0]],[self.atp[tag][i][j][1],self.tpoints[tag][i][1]])
                            # plt.plot(self.acp[tag][i][j][0] + np.cos(theta) * self.arad[tag][i][j],self.acp[tag][i][j][1] + np.sin(theta) * self.arad[tag][i][j])
                            plt.scatter(self.acp[tag][i][j][0],self.acp[tag][i][j][1],5,color='purple')
                            plt.scatter(self.atp[tag][i][j][0],self.atp[tag][i][j][1],5,color='red')
                            plt.scatter(self.tpoints[tag][i][0],self.tpoints[tag][i][1],5,color='blue')
                            plt.title('{},radius : {}, distance : {}'.format(i,self.arad[tag][i][j],getDistance(self.tpoints[tag][i],self.atp[tag][i][j])))
                            
                            plt.savefig(path + 'fig{:04d}.png'.format(svnum))
                            svnum += 1
                            j += 1
                        sx.append(self.acp[tag][i][j - 1][0])
                        sy.append(self.acp[tag][i][j - 1][1])
                        i += 1
                        if i == stop:
                            break
                    tag += 1
                
            elif mode[index] == 3:
                pt = []
                plt.clf()
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                i = 0
                tx = []
                ty = []
                while i < len(self.IntPoints):
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                plt.scatter(tx,ty,5)
                plt.scatter(0.9,0.7)
                tx = []
                ty = []
                tx.append(0.9)
                ty.append(0.7)
                tx.append(0.9 + 1000 * -np.cos(3 * np.pi / 8))
                tx.append(0.9 + 1000 * -np.cos(np.pi / 8))
                ty.append(0.7 + 1000 * -np.sin(3 * np.pi / 8))
                ty.append(0.7 + 1000 * -np.sin(np.pi / 8))
                plt.plot([tx[0],tx[1]],[ty[0],ty[1]])
                plt.plot([tx[0],tx[2]],[ty[0],ty[2]])
                i = 0
                tx = []
                ty = []
                theta = np.linspace(0, np.pi * 2)
                while i < len(pt):
                    tx.append(pt[i][0])
                    ty.append(pt[i][1])
                    plt.scatter(tx[i],ty[i],5)
                    # plt.plot(tx[i] + r[i] * np.cos(theta),ty[i] + r[i] * np.sin(theta),5)
                    i += 1
                plt.savefig('SearchRecovery.png')
            elif mode[index] == 4:
                theta =  np.linspace(0,2*np.pi,100)
                #This is the figure which can display the quadtree along with its nodes
                plt.clf()
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                i = 0
                tx = []
                ty = []
                plt.title('threshold:{}'.format(self.threshDistance[0]))
                while i < len(self.IntPoints):
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                plt.scatter(tx,ty,5)
                i = 0
                while i < len(self.Strees):
                    self.Strees[i].plot(theta)
                    i+= 1
                plt.savefig('nodes.png')
            elif mode[index] == 5:
                #Plots Connections of the Ordering
                plt.clf()
                plt.xlim(xmin,xmax)
                plt.ylim(ymin,ymax)
                i = 0
                tx = []
                ty = []
                while i < len(self.IntPoints):
                    tx.append(self.IntPoints[i][0])
                    ty.append(self.IntPoints[i][1])
                    i += 1
                plt.scatter(tx,ty,5)
                #plot the sections
                tx = []
                ty = []
                tr = []
                i = 0
                while i < len(self.orderpoints):
                    j = 0
                    while j < len(self.orderpoints[i]):
                        tx.append(self.orderpoints[i][j][0][0])
                        ty.append(self.orderpoints[i][j][0][1])
                        tr.append(self.orderpoints[i][j][1])
                        j += 1
                    i += 1
                sc = plt.scatter(tx,ty,5,c=tr,cmap='winter')
                plt.colorbar(sc)
                i = 0
                while i < len(self.delpoints):
                    plt.scatter(self.delpoints[i][0],self.delpoints[i][1],10,color='black')
                    i += 1
                plt.show()
                plt.savefig('orderLines.png')
            et = time.time()
            tt += (et - st)
            index += 1        
        eet = time.time()
        tttt = eet - sst
        print('abstime {} minuites {} seconds'.format(tttt // 60, tttt % 60))
        print('Animation took {} minuites and {} seconds'.format((tt) // 60,(tt) % 60))
                
    def savedat(self,mode : int = 0):
        print('Saving ... Please Wait')
        st = time.time()
        i = 0
        tx = []
        ty = []
        tz = []
        tr = []
        if mode == 0:
            while i < len(self.orderpoints):
                j = 0
                while j < len(self.orderpoints[i]):
                    tx.append(self.orderpoints[i][j][0][0])
                    ty.append(self.orderpoints[i][j][0][1])
                    if self.dim == 3:
                        tz.append(self.orderpoints[i][j][0][2])
                    tr.append(self.orderpoints[i][j][1])
                    j += 1
                i += 1
        elif mode == 1:
            while i < len(self.SkelePoints):
                j = 0
                while j < len(self.SkelePoints[i]):
                    tx.append(self.SkelePoints[i][j][0])
                    ty.append(self.SkelePoints[i][j][1])
                    if self.dim == 3:
                        tz.append(self.SkelePoints[i][j][2])
                    tr.append(self.SkeleRad[i][j])
                    j += 1
                i += 1
        if self.dim == 2:
            output = pd.DataFrame({'x':tx,'y':ty,'r':tr})
        else:
            output = pd.DataFrame({'x':tx,'y':ty,'z':tz,'r':tr})
        output[1:].to_csv('SkeleSave.dat',index=False)
        et = time.time()
        tt = et - st
        print('Save Complete! Took {} Minuites and {} Seconds'.format(tt // 60, tt % 60))
