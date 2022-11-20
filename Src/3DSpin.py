import csv
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os
import sys
import getopt
import numpy as np
from scipy import interpolate as itp
from pathos.pools import ProcessPool
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)) + r'/DataStruct')
from DataStructures import quicksort 
def paraplot(data,start : int,stop : int):
    import matplotlib.pyplot as plt
    tx = data[0]
    ty = data[1]
    tz = data[2]
    if data[-1] == 0:
        tr = data[3]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #if data[0][len(data[0]) - 1] == 0:
    #    p = ax.scatter3D(tx,ty,tz,s=5,c=tr,cmap='rainbow')
    #else:
    if data[-1] == 0:
        p = ax.scatter3D(tx,ty,tz,s=12,c=tr,cmap='rainbow')
    else:
        p = ax.scatter3D(tx,ty,tz,s=12,c=tz,cmap='rainbow')
    fig.colorbar(p)
    i = start
    while i < stop:
        print('step',i)
        ax.view_init(30,i)
        save = source + r'AnimationData/Spin/{0:0=3d}spin.png'.format(i)
        plt.savefig(save)
        #print(i)
        i += 1

if __name__ == '__main__':
    
    #Gets Args, Only Arg is number of processes though
    argumentList = sys.argv[1:]
    options = "i:n:m:"
    nodes = 1
    fp = r''
    mode = 0
    try:
        arguments,argumentList = getopt.getopt(argumentList,options) 
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-i"):
                fp = str(currentValue)
            if currentArgument in ("-n"):
                nodes = int(currentValue)
            if currentArgument in ("-m"):
                mode = int(currentValue)
                #Default mode = 0 => Skeledat with rad
                #mode 1 => Interface data
    except getopt.error as err:
        print(str(err))

    #Loads Data from results
    plt.rcParams['figure.dpi'] = 300
    source = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + r'/'
    if len(fp) == 0:
        inpath = source + r'SkeleData/Input/infc_50.dat'
    else:
        inpath = source + r'SkeleData/' + fp
    if mode == 0:
        with open(inpath,'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ',')
            tx = []
            ty = []
            tz = []
            tr = []
            liner = []
            i = 0
            j = 0
            for row in data:
                if str(row[0]) == 'x':
                    a = 0
                elif float(row[3]) > -100000000:
                    i += 1
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    r = float(row[3])
                    if j % 25 == 0:
                        tx.append(x)
                        ty.append(y)
                        tz.append(z)
                        tr.append(r)
                        #Save x=0 data here for sep figure
                    if y > -0.1 and y < 0.1:
                        liner.append([z,r])
                    j += 1
    elif mode == 1:
        with open(inpath,'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ' ')
            tx = []
            ty = []
            tz = []
            i = 0
            for row in data:
                print(i,row)
                if str(row[0]) == '#1:x':
                    a = 0
                else:
                    i += 1
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    tx.append(x)
                    ty.append(y)
                    tz.append(z)
    if isinstance(liner,list) and len(liner) > 0:
        plt.clf()
        fig = plt.figure()
        saveapprox = source + 'Plot/radapprox.png'
        newdata = quicksort(liner,0)
        plotz = [] 
        plotr = []
        j = 0
        for p in newdata:
            if j % 3 == 0 and (len(plotz) ==  0 or p[0] != plotz[len(plotz)-1]):
                plotz.append(p[0])
                plotr.append(p[1])
            j += 1
        plt.plot(plotz,plotr,color='blue')
        plt.savefig(saveapprox)
    if nodes > 1:
        i = 0
        st = []
        sp = []
        if nodes > 30:
            nodes = 30
        nf = int(np.floor(360/nodes))
        nc = int(np.ceil(360/nodes))
        last = 0
        data = []
        md = []
        while i < nodes and last + nc < 360:
            st.append(last)
            if i % 2 == 0:
                sp.append(last + nf)
                last = last + nf 
            else:
                sp.append(last + nc)
                last = last + nc
            if mode == 0:
                data.append([tx,ty,tz,tr,mode])
            elif mode == 1:
                data.append([tx,ty,tz,mode])
            i += 1
        sp[len(sp) - 1] = 360
        pool = ProcessPool(nodes=nodes)
        pool.map(paraplot,data,st,sp)
    else:
        if mode == 0:
            paraplot([tx,ty,tz,tr,0],0,360)
        else:
            paraplot([tx,ty,tz,1],0,360)

