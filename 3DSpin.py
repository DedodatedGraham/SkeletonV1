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
        elif float(row[3]) > -100000000:
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
fig = plt.figure()
ax = plt.axes(projection='3d')
p = ax.scatter3D(tx,ty,tz,s=5,c=tr,cmap='winter')
fig.colorbar(p)
i = 0
while i < 360:
    ax.view_init(30,i)
    plt.savefig('AnimationData/Spin/{0:0=3d}spin.png'.format(i))
    print(i)
    i += 1
