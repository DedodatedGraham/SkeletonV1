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
with open('SkeleSave.dat','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ',')
    tx = []
    ty = []
    tz = []
    tr = []
    i = 0
    for row in data:
        if str(row[0]) == 'x':
            a = 0
        else:
            i += 1
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            r = float(row[3])
            print(i)
            tx.append(x)
            ty.append(y)
            tz.append(z)
            tr.append(r)
                        
#with open('vof_points_norm_0650.dat','r') as csvfile:
#    data = csv.reader(csvfile, delimiter = ' ')
#    ttx = []
#    tty = []
#    ttz = []
#    tnx = []
#    tny = []
#    tnz = []
#    i = 0
#    for row in data:
#        if str(row[0]) == 'x':
#            a = 0
#        elif len(row) == 6:
#            if float(row[0]) > 0 and float(row[1]) > 0 and float(row[2]) > 0:
#                print(i)
#                i += 1
#                ttx.append(float(row[0]))
#                tty.append(float(row[1]))
#                ttz.append(float(row[2]))
#                tnx.append(float(row[3]))
#                tny.append(float(row[4]))
#                tnz.append(float(row[5]))

#with open('vof_points_norms.dat','r') as csvfile:
#    data = csv.reader(csvfile,delimiter = ' ')
#    txx = []
#    tyy = []
#    nxx = []
#    nyy = []
#    i = 0
#    for row in data:
#        if str(row[0]) == 'x':
#            a = 0
#        else:
#            txx.append(float(row[0]))
#            tyy.append(float(row[1]))
#            nxx.append(float(row[2]))
#            nyy.append(float(row[3]))
#fig = plt.figure()
#plt.scatter(tx,ty,5,color='blue')
#plt.plot(,5,color='green')
#plt.savefig('2DInt.png')

#Next We Do the Actual Plotting 
fig = plt.figure()
ax = plt.axes(projection='3d') 
p = ax.scatter3D(tx,ty,tz,s=5,c=tr,cmap='winter')
fig.colorbar(p)
plt.savefig('Skeleton3DFIG.png')
#plt.clf()
#q = ax.scatter3D(ttx,tty,ttz,s=5,c=ttz,cmap='viridis')
#fig.colorbar(q)
#ax.plot3D([ttx[1000],ttx[1000] + tnx[1000]],[tty[1000],tty[1000] + tny[1000]],[ttz[1000],ttz[1000] + tnz[1000]])
#plt.savefig('3DInterface.png')
