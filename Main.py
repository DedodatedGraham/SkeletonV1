#plotting
from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import Skeletize
from DataStructures import kdTree

TestPoints = []
NormPoints = []
#loads in test case
with open('interface_points_020000.dat','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ' ')
    for row in data:
        TestPoints.append([float(row[0]),float(row[1])])
        NormPoints.append([float(row[2]),float(row[3])])
    csvfile.close()




#when making the tree it changes test points?
testTree = kdTree(TestPoints,2)
q = [[0.5,1],[1,0]]
a = testTree.treeLines2D(q)
finPoints,finR = Skeletize.Skeletize2D(TestPoints, NormPoints)
testX = []
testY = []
finX = []
finY = []
i = 0
while i < len(TestPoints):
    testX.append(TestPoints[i][0])
    testY.append(TestPoints[i][1])
    i = i + 1
    
    
# i = 0
# while i < len(a):
#     x = [a[i][0],a[i+1][0]]
#     y = [a[i][1],a[i+1][1]]
#     plt.plot(x,y)
#     i = i + 2

i = 0
# while i < len(finPoints):
#     finX.append(finPoints[i][0])
#     finY.append(finPoints[i][1])
#     i = i + 1
# plt.scatter(testX, testY)
# plt.scatter(finX, finY)


# i = 0
# theta = np.linspace(0, 2*np.pi, 100)
# while i < 100:
#     r = finR[i]
#     x1 = finPoints[i][0] + r*np.cos(theta)
#     x2 = finPoints[i][1] + r*np.sin(theta)
#     plt.plot(x1, x2)
#     i = i + 1


