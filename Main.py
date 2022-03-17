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
with open('interface_points_070000.dat','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ' ')
    for row in data:
        TestPoints.append([float(row[0]),float(row[1])])
        NormPoints.append([float(row[2]),float(row[3])])
    csvfile.close()

#x = [[0,0],[0,1],[0,2.4],[1.1,2],[2,2],[1.1,0],[0.3,2],[10,20],[4.5,6],[-3.2,1.4],[6,-2],[3.2,1],[0,-1.6],[-7,8]]
#TestTree = kdTree(x, 2)
#found = TestTree.getNear(0.4, 2,[0,0])
#print(found)
# print(TestTree.tree[0])
# print(TestTree.tree[1])
# print(TestTree.tree[2])
TestPoint = TestPoints[randint(0,len(TestPoints))]

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
i = 0
while i < len(finPoints):
    finX.append(finPoints[i][0])
    finY.append(finPoints[i][1])
    i = i + 1
plt.scatter(testX, testY)
plt.scatter(finX, finY)
# i = 0
# while i < len(x):
    # finX.append(x[i][0])
    # finY.append(x[i][1])
    # i = i + 1
    
# plt.scatter(finX,finY)