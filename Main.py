#plotting
from random  import randint
from sys import float_repr_style
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import Skeletize

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

