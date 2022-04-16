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

#ellipse, norm isnt working well
i = 0
theta = np.linspace(0, 2*np.pi, 100)
while i < 100:
    r = 0.5
    TestPoints.append([0.5 + r*np.cos(theta[i]),0.5 + 2*r*np.sin(theta[i])])
    i = i + 1
    
i = 0
while i < 100:
    if i == 99:
        a = [TestPoints[i - 1][0] - TestPoints[i][0],TestPoints[i - 1][1] - TestPoints[i][1]]
        b = [TestPoints[0][0] - TestPoints[i][0],TestPoints[0][1] - TestPoints[i][1]]
    else:
        a = [TestPoints[i - 1][0] - TestPoints[i][0],TestPoints[i - 1][1] - TestPoints[i][1]]
        b = [TestPoints[i + 1][0] - TestPoints[i][0],TestPoints[i + 1][1] - TestPoints[i][1]]
    c = [a[0] + b[0],a[1] + b[1]]
    NormPoints.append([-c[0] + TestPoints[i][0],-c[1] + TestPoints[i][1]])
    i = i + 1
   

#square with norm working
# i = 0
# while i < 100:
#     if i < 25:
#         TestPoints.append([0.0, i / 25])
#         if i != 0:
#             NormPoints.append([-1.0,0.0])
#         else:
#             NormPoints.append([-1.0,-1.0]) 
#     elif i > 24 and i < 50:  
#         TestPoints.append([(i % 25) / 25,1.0])
#         if i != 25:
#             NormPoints.append([0.0,1.0])
#         else:
#             NormPoints.append([-1.0,1.0])
#     elif i > 49 and i < 75:
#         TestPoints.append([1.0,(25-(i % 25)) / 25])
#         if i != 50:
#             NormPoints.append([1.0,0.0])
#         else:
#             NormPoints.append([1.0,1.0])
#     else:
#         TestPoints.append([(25 - (i % 25)) / 25,0.0])
#         if i != 75:
#             NormPoints.append([0.0,-1.0])
#         else:
#             NormPoints.append([1.0,-1.0])
#     i = i + 1

#half circle half square
# i = 0
# r = 0.5
# theta = np.linspace((3/2)*np.pi, (5/2)*np.pi, 50)
# while i < 100:
#     if i < 25:
#         TestPoints.append([0.0, i / 25])
#         if i != 0:
#             NormPoints.append([-1.0,0.0])
#         else:
#             NormPoints.append([-1.0,-1.0]) 
#     elif i > 24 and i < 37:  
#         TestPoints.append([(i % 25) / 25,1.0])
#         if i != 25:
#             NormPoints.append([0.0,1.0])
#         else:
#             NormPoints.append([-1.0,1.0])
#     elif i > 88:
#         TestPoints.append([(25 - (i % 25)) / 25,0.0])
#         if i != 75:
#             NormPoints.append([0.0,-1.0])
#         else:
#             NormPoints.append([1.0,-1.0])
#     else:
#         if i == 60:
#             TestPoints.append([0.5 + r*np.cos(theta[49 - i % 50]),0.5 + 1.1*r*np.sin(theta[49 - i % 50])])
#         else:
#             TestPoints.append([0.5 + r*np.cos(theta[49 - i % 50]),0.5 + r*np.sin(theta[49 - i % 50])])
#         NormPoints.append([TestPoints[i][0] - 0.5,TestPoints[i][1]-0.5])
#     i = i + 1



# loads in test case
# with open('interface_points_020000.dat','r') as csvfile:
#     data = csv.reader(csvfile, delimiter = ' ')
#     for row in data:
#         TestPoints.append([float(row[0]),float(row[1])])
#         NormPoints.append([float(row[2]),float(row[3])])
#     csvfile.close()




#when making the tree it changes test points?
finPoints,finR = Skeletize.Skeletize2D(TestPoints, NormPoints)
# print(finPoints,finR)
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
while i < len(finPoints):
    finX.append(finPoints[i][0])
    finY.append(finPoints[i][1])
    plt.plot([testX[i],finX[i]],[testY[i],finY[i]])
    i = i + 1
plt.scatter(testX, testY)
plt.scatter(finX, finY)




plt.xlim(-0.1,1.1)
plt.ylim(-0.8,1.8)
