import csv
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
#Loads Data from results
fig = plt.figure()
ax = plt.axes(projection='3d')
fac = 1
centers = [[-69.77266668149157, -35.78270185421122, -55.82423207373445], [-13.96363124587302, -6.676643295672345, -10.992079955624154], [-13.96363124587302, -6.676643295672345, -10.992079955624154]]
testp = [[0.18661, 0.703125, 0.375], [0.1875, 0.703125, 0.373892], [0.1875, 0.703125, 0.373892]]
rads= [96.87037545457554, 19.593387, 19.593387]
point =[0.18661, 0.703125, 0.375]
norm = [0.7221947510082366, 0.37664587014344914, 0.5801487999815527]
xline = [point[0],point[0] + fac * norm[0]]
yline = [point[1],point[1] + fac * norm[1]]
zline = [point[2],point[2] + fac * norm[2]]
i = 0
dx = []
dy = []
dz = []
while i < len(centers):
    dx.append(centers[i][0])
    dy.append(centers[i][1])
    dz.append(centers[i][2])
    dx.append(testp[i][0])
    dy.append(testp[i][1])
    dz.append(testp[i][2])
    i += 1
with open('../disk1.dat','r') as csvfile:
    data = csv.reader(csvfile, delimiter = ' ')
    tx = []
    ty = []
    tz = []
    i = 0
    for row in data:
        if i == 0:
            i = 0
        else:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            print(i)
            tx.append(x)
            ty.append(y)
            tz.append(z)
        i += 1
#print(tz)
tz = np.array(tz)
print(tz)
#print(tx)
#print(ty)
ax.scatter3D(tx,ty,tz,c='green',s=1)
ax.plot3D(xline,yline,zline,c='blue')
#ax.scatter3D(dx,dy,dz,c='orange')
plt.savefig('skeletest.png')
