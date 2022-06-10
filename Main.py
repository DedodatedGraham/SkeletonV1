from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import numpy as np
import csv
import scipy
import pandas as pd
import os
import time

import Skeletize
import DataStructures 
from DataStructures import kdTree
from SkeleNet import SkeleNet

plt.rcParams['figure.dpi'] = 300

theta = np.linspace(0,2*np.pi,100)

xmin = -0.3
xmax = 0.2
ymin = -0.3
ymax = 0.2

tstart = time.time()
IntPoints = []
TestPoints = []
NormPoints = []

fig = plt.figure()
ax = plt.subplot(111)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

###TEST CASES ONLY ONE SHOULD BE ACTIVE AT A TIME:

# loads in test case from data (020000.dat or 070000.dat)
i = 0
with open('spiral.dat','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ' ')
    for row in data:
        if randint(0,10) >= 0:
            IntPoints.append([float(row[0]),float(row[1])])
            NormPoints.append([float(row[2]),float(row[3])])
    csvfile.close()
print(NormPoints[0])
NormPoints = DataStructures.normalize2D(NormPoints)
net = SkeleNet(IntPoints)

for point in IntPoints:
    if point == IntPoints[0]:
        print('whoops')
print(IntPoints[0],NormPoints[0])
print(net.IntPoints[0],net.NormPoints[0])


#testX = []
#testY = []
#i = 0
#while i < len(TestPoints):
#    testX.append(TestPoints[i][0])
#    testY.append(TestPoints[i][1])
#    # testZ.append(TestPoints[i][2])
#    i = i + 1
####SOLVING THE POINTS
#
#TskeleStart = time.time()
#finPoints,finR,animfile,animdfile = Skeletize.Skeletize2D(TestPoints, NormPoints,0 , len(TestPoints))
## finPoints,finR = Skeletize.Skeletize3D(TestPoints, NormPoints) 
#TskeleEnd = time.time()
#print("Skeletinzation time",TskeleEnd - TskeleStart,"seconds")
#
####PROCESSING DATA INTO FIGURES
#finX = []
#finY = []
## finz = []
#intX = []
#intY = []
#
#
#    
#    
## i = 0
## while i < len(a):
##     x = [a[i][0],a[i+1][0]]
##     y = [a[i][1],a[i+1][1]]
##     plt.plot(x,y)
##     i = i + 2
#
#i = 0
## while i < len(IntPoints):
##     intX.append(IntPoints[i][0])
##     intY.append(IntPoints[i][1])
##     i += 1
#
#    
#
#i = 0
#while i < len(finPoints):
#    finX.append(finPoints[i][0])
#    finY.append(finPoints[i][1])
#    # finZ.append(finPoints[i][2])
#    #if you want to see final circles in a figure
#    # theta = np.linspace(0,2*np.pi,100)
#    # q = 0
#    # tx = []
#    # ty = []
#    # if i > 20 and i < 40:
#    #     while q < 100:
#    #         tx.append(finX[i] + finR[i]* np.cos(theta[q]))
#    #         ty.append(finY[i] + finR[i]* np.sin(theta[q]))
#    #         q = q + 1
#    #     plt.plot(tx,ty)
#    i += 1
#
#
##For unfiltered results
## finX = []
## finY = []    
## i = 0
## while i < len(animfile[0]):
##     finX.append(animfile[0][i][len(animfile[0][i]) - 1][0])
##     finY.append(animfile[0][i][len(animfile[0][i]) - 1][1])
##     i += 1
#    
#    
#plt.clf()
#plt.xlim(xmin,xmax)
#plt.ylim(ymin,ymax)
#plt.scatter(testX, testY, zorder = 1)
#plt.scatter(finX, finY, zorder = 2)
## i = 0
## while i < len(finX):
##     plt.plot(finX[i] + np.cos(theta) * finR[i],finY[i] + np.sin(theta) *finR[i])
##     i += 1
#plt.savefig("Output.png")
## ax = plt.axes(projection='3d')
#
####ANIMATED FIGURE(ANIMFILE holds all information needed for building the animated figures)
#
##taniStart = time.time()
##
##
##plt.clf()
##cx = []
##cy = []
##target = 0
##countt = 0
##countc = 0
##calc = 0
##temp = []
##i = 0  
##
##
##while i < len(animfile[0]):
##    calc = calc + len(animfile[0][i])
##    temp.append(len(animfile[0][i]))
##    i = i + 1
##
##
##
##
##print("length",len(animfile[0]))
##print("Building Animation...")
##p = 0
##case = True
##while case:
##    if not(os.path.isdir(os.getcwd() + "/AnimationData/{:04d}".format(p))):
##        os.mkdir(os.getcwd() + "/AnimationData/{:04d}".format(p))
##        case = False
##    else:
##        p +=1
##     
##     
##i = 0 
##count = 0
###Building
##while i < len(animfile[0]):
##    j = 0
##    while j < len(animfile[0][i]):
##        plt.clf()
##        plt.xlim(xmin,xmax)
##        plt.ylim(ymin,ymax)
##        #Interface
##        plt.scatter(intX,intY,marker = "X",color = "black")
##        plt.scatter(testX,testY,marker = "X",color = "yellow")
##        print(i,"/",len(temp) - 1 , j , '/', temp[i] - 1)
##        #curent norm
##        tx = []
##        ty = [] 
##        tx.append(animfile[3][i][0] + animfile[4][i][0] * 1000)
##        tx.append(animfile[3][i][0] + animfile[4][i][0] * -1000)
##        ty.append(animfile[3][i][1] + animfile[4][i][1] * 1000)
##        ty.append(animfile[3][i][1] + animfile[4][i][1] * -1000)
##        plt.plot(tx,ty)
##      
##        #Plot temporary varibles(change per frame)
##        tx = []
##        ty = []
##        tx.append(animfile[1][i][j-1][0])
##        ty.append(animfile[1][i][j-1][1])
##        plt.scatter(tx,ty,color = "blue", marker = "o")
##      
##        tx = []
##        ty = []
##        tx.append(animfile[0][i][j][0])
##        ty.append(animfile[0][i][j][1])
##        plt.scatter(tx,ty,color = "purple", marker = "o")
##      
##      
##      
##        plt.plot(animfile[2][i][j] * np.cos(theta) + tx[0],animfile[2][i][j] * np.sin(theta) + ty[0]) 
##        #plotting main point                                                           
##        plt.scatter(animfile[3][i][0],animfile[3][i][1],color = "green", marker = 's')
##        if j == len(animfile[0][i]) - 1:
##            cx.append(tx[0])
##            cy.append(ty[0])
##     
##      
##      
##
##        #Plotting for the saved calculated centerpoints
##        if not(len(cx) == 0):
##            plt.scatter(cx,cy,color = "red", marker = "^")
##      
##        tx = []
##        ty = []
##        tx.append(animfile[0][i][j][0])
##        ty.append(animfile[0][i][j][1])
##        tx.append(animfile[3][i][0])
##        ty.append(animfile[3][i][1])
##      
##        plt.plot()
##        #formatting
##        plt.title("Current r={}; Current Centerpoint[{},{}]".format(animfile[2][i][j],animfile[0][i][j][0],animfile[0][i][j][1]), size = 8)
##        plt.xlabel("TestPoint: {} ; TreePoint: {}".format(animfile[3][i],animfile[1][i][j-1]))
##      
##        save = os.getcwd() + "/AnimationData/{:04d}/fig{:04d}.png".format(p,count)
##        plt.savefig(save) 
##      
##
##
##        count += 1
##        j += 1
##    i += 1
##  
###thinning
##count2 = 0
##i = 0
##x = []
##y = []
##while i < len(animdfile):
##    print(i,"/",len(animdfile)-1)
##    x.append(animdfile[i][0])
##    y.append(animdfile[i][1])
##    plt.clf()
##    plt.xlim(xmin,xmax)
##    plt.ylim(ymin,ymax)
##    #Interface
##    plt.scatter(intX,intY,marker = "X",color = "black")
##    plt.scatter(testX,testY,marker = "X",color = "yellow")
##    #Plotting for the saved calculated centerpoints
##    if not(len(cx) == 0):
##        plt.scatter(cx,cy,color = "blue", marker = "^")
##    plt.scatter(x, y,color = "red", marker = "X")
##  
##    plt.title("Deleted:{}/{}".format(i,len(animdfile)-1),size = 8)
##    save = os.getcwd() + "/AnimationData/{:04d}/fig{:04d}.png".format(p,count + count2)
##    plt.savefig(save)
##    count2 += 1
##    i += 1
##
##count3 = 0
##
##
##
##
##plt.clf()
##plt.xlim(xmin,xmax)
##plt.ylim(ymin,ymax)
####Interface
##plt.scatter(intX,intY,marker = "X",color = "black")
##plt.scatter(testX,testY,marker = "X",color = "yellow")
##
##i = 0
##while i < len(finX):
##    print(i,"/",len(finX)-1)
##    plt.scatter(finX[i],finY[i],color = "blue", marker = "^")
##    plt.plot(finX[i] + finR[i] * np.cos(theta), finY[i] + finR[i] * np.sin(theta))
##      
##    save = os.getcwd() + "/AnimationData/{:04d}/fig{:04d}.png".format(p,count + count2 + count3)
##    plt.savefig(save)
##  
##    count3 += 1
##    i += 1
##  
##taniEnd = time.time()
##print('Animation took {} minuites and {} seconds'.format((taniEnd-taniStart) // 60,(taniEnd-taniStart) % 60))
