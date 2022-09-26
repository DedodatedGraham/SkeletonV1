import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import sys
import getopt
import numpy as np
from pathos.pools import ProcessPool

def paraplot(data,start : int,stop : int):
    tx = data[0]
    ty = data[1]
    tz = data[2]
    tr = data[3]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    p = ax.scatter3D(tx,ty,tz,s=5,c=tr,cmap='winter')
    fig.colorbar(p)
    i = start
    while i < stop:
        ax.view_init(30,i)
        save = source + r'AnimationData/Spin/{0:0=3d}spin.png'.format(i)
        plt.savefig(save)
        #print(i)
        i += 1

if __name__ == '__main__':
    
    #Gets Args, Only Arg is number of processes though
    argumentList = sys.argv[1:]
    options = "n:"
    nodes = 1
    try:
        arguments,argumentList = getopt.getopt(argumentList,options) 
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-n"):
                nodes = int(currentValue)
    except getopt.error as err:
        print(str(err))

    #Loads Data from results
    plt.rcParams['figure.dpi'] = 300
    source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/'
    inpath = source + r'SkeleData/Output/SkeleSave.dat'
    with open(inpath,'r') as csvfile:
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
                tx.append(x)
                ty.append(y)
                tz.append(z)
                tr.append(r)
    i = 0
    st = []
    sp = []
    if nodes > 30:
        nodes = 30
    nf = int(np.floor(360/nodes))
    nc = int(np.ceil(360/nodes))
    last = 0
    data = []
    while i < nodes and last + nc < 360:
        st.append(last)
        if i % 2 == 0:
            sp.append(last + nf)
            last = last + nf 
        else:
            sp.append(last + nc)
            last = last + nc
        data.append([tx,ty,tz,tr])
        i += 1
    sp[len(sp) - 1] = 360

    pool = ProcessPool(nodes=nodes)
    pool.map(paraplot,data,st,sp)
