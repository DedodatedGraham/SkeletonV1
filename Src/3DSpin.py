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

def paraplot(data,start : int,stop : int,ids : int):
    print("Beggining Plot")
<<<<<<< HEAD
    if data[-1] == 0 or data[-1] == 1:
        import matplotlib
        import matplotlib.pyplot as plt
    else:
        from mayavi import mlab
=======
    import matplotlib.pyplot as plt
>>>>>>> ed1b910f9eabacd8ab874e9d67afcd381d8443dc
    #print('importing data')
    tx = data[0]
    ty = data[1]
    tz = data[2]
    if data[-1] == 0 or data[-1] == 2:
        tr = data[3]
    if data[-1] == 2:
        intx = data[4]
        inty = data[5]
        intz = data[6]
    #print('Loaded Data',data[-1])
    fig = plt.figure()
    #print('got 2dfig')
    ax = plt.axes(projection='3d')
    ax.axes.set_xlim3d(left=-0.7,right=0.7)
    ax.axes.set_ylim(bottom=-0.7,top=0.7)
    ax.axes.set_zlim3d(bottom=-0.7,top=0.7)
    #print('Loaded fig')
    if data[-1] == 0:
        #Draw skeleton Points
<<<<<<< HEAD
        i = 0
        txx = []
        tyy = []
        tzz = []
        cmap = matplotlib.cm.rainbow
        norm = matplotlib.colors.Normalize(vmin =min(tr),vmax=max(tr))
        colorbar = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap);
        while(i < len(tx)):
            u,v = np.mgrid[0.0:np.pi:20j,0.0:2*np.pi:20j]
            txx = tx[i] + tr[i] * np.cos(u) * np.sin(v)
            tyy = ty[i] + tr[i] * np.sin(u) * np.sin(v)
            tzz = tz[i] + tr[i] * np.cos(v)
            j = 0
            color = []
            while(j < 400):
                color.append(tr[i])
                j += 1
            ax.scatter3D(txx,tyy,tzz,s=10,c=color,cmap=cmap,norm=norm)
            i += 1
        fig.colorbar(colorbar,ax=ax)
=======
        p = ax.scatter3D(tx,ty,tz,s=5,c=tr,cmap='rainbow')
        fig.colorbar(p)
>>>>>>> ed1b910f9eabacd8ab874e9d67afcd381d8443dc
    elif data[-1] == 2:
        p = ax.scatter3D(tx,ty,tz,c=tr,cmap='rainbow')
        fig.colorbar(p)
        q = ax.scatter3D(intx,inty,intz,s=5,c=tz,cmap='Greys',alpha=0.5)
    else:
        p = ax.scatter3D(tx,ty,tz,s=15,c=tz,cmap='Greys')
        fig.colorbar(p)

    #Rotate & save
    print('starting animation')
    i = start
    while i < stop:
        print('step',i)
        ax.view_init(30,i)
        print('Rotated', i)
        save = source + r'AnimationData/Spin/{0:0=3d}spin.png'.format(i)
        print('Wrote save name',i)
        plt.savefig(save)
        print('Saved',i)
        i += 1
####MAIN
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
    print(mode,"mode try")
    tx = []
    ty = []
    tz = []
    tr = []
    if mode == 0 or mode == 2:
        with open(inpath,'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ' ')
            liner = []
            i = 0
            j = 0
            for row in data:
                print('loading',i,j)
                if str(row[0]) == 'x':
                    a = 0
                elif float(row[3]) > -10000000000:
                    i += 1
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    r = float(row[3])
                    tx.append(x)
                    ty.append(y)
                    tz.append(z)
                    tr.append(r)
                    j += 1
    if mode == 2:
        #Gets interface adress if mode 2
        inpath =  source + r'SkeleData/Input/infc_50.dat';
    txx = []
    tyy = []
    tzz = []
    if mode == 1 or mode == 2:
        with open(inpath,'r') as csvfile:
            data = csv.reader(csvfile, delimiter = ' ')
            i = 0
            for row in data:
                print(i,row)
                if str(row[0]) == '#1:x' or str(row[0]) == 'x':
                    a = 0
                else:
                    i += 1
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])
                    txx.append(x)
                    tyy.append(y)
                    tzz.append(z)
    if mode == 0 and isinstance(liner,list) and len(liner) > 0:
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
        fit = np.polyfit(plotz,plotr,3)
        plt.plot(plotz,plotr,color='blue')
        plt.savefig(saveapprox)
    print('building')
    if nodes > 1:
        i = 0
        st = []
        sp = []
        ids = []
        if nodes > 30:
            nodes = 30
        nf = int(np.floor(360/nodes))
        nc = int(np.ceil(360/nodes))
        last = 0
        data = []
        md = []
        while i < nodes and last + nc < 360:
            ids.append(i)
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
                data.append([txx,tyy,tzz,mode])
            elif mode == 2:
                data.append([tx,ty,tz,tr,txx,tyy,tzz,mode])
            i += 1
        sp[len(sp) - 1] = 360
        pool = ProcessPool(nodes=nodes)
        pool.map(paraplot,data,st,sp,ids)
    else:
        if mode == 0:
            paraplot([tx,ty,tz,tr,mode],0,360,0)
        elif mode == 1:
            paraplot([tx,ty,tz,mode],0,360,0)
        else:
            paraplot([tx,ty,tz,tr,txx,tyy,tzz,mode],0,360,0)
