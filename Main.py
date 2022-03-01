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
    finY.append(finPoints[i][0])
    i = i + 1
plt.scatter(testX, testY)
plt.scatter(finX, finY)