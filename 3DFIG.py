# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:32:29 2022

@author: graha
"""

import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#Loads Data from results
plt.rcParams['figure.dpi'] = 300
#with open('SkeleSave.dat','r') as csvfile:
#    data = csv.reader(csvfile, delimiter = ',')
#    tx = []
#    ty = []
#    tz = []
#    tr = []
#    i = 0
#    for row in data:
#        if str(row[0]) == 'x':
#            a = 0
#        else:
#            # print(i)
#            i += 1
#            x = float(row[0])
#            y = float(row[1])
#            z = float(row[2])
#            r = float(row[3])
#            print(x,y,z,r)
#            if x > 2.8 and x <  3.6:    
#                if y > 0.0 and y <  1.0:
#                    if z > 0.0 and z <  1.0:
#                        if r < 0.1 and r > 0.0045:
#                            tx.append(x)
#                            ty.append(y)
#                            tz.append(z)
#                           tr.append(r)
                        
with open('vof_points_norm_0650.dat','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ' ')
    ttx = []
    tty = []
    ttz = []
    i = 0
    for row in data:
        print(i)
        if str(row[0]) == 'x':
            a = 0
        elif len(row) == 6:
            print(i)
            i += 1
            print(row)
            if not(int(row[6]) < 0):
                ttx.append(float(row[0]))
                tty.append(float(row[1]))
                ttz.append(float(row[2]))

#Next We Do the Actual Plotting 
fig = plt.figure()
ax = plt.axes(projection='3d') 
#p = ax.scatter3D(tx,ty,tz,s=5,c=tr,cmap='winter')
#fig.colorbar(p)
q = ax.scatter3D(ttx,tty,ttz,s=5,c=ttz,cmap='viridis')
fig.colorbar(q)
plt.savefig('3DPlot.png')
