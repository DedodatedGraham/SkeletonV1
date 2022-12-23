#!/bin/sh
#PBS -l nodes=1:ppn=30
#PBS -o ../Log/3D.out
#PBS -e ../Log/3D.err

cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`

#mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/Load.py
#mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/3DSpin.py -i 'Input/infc_0.dat' -n 50 -m 1
#mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/3DSpin.py -i 'Output/infc_50_savePrePurge.dat' -n 50 -m 0
mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/3DSpin.py -i 'Output/infc_50_savePrePurge.dat' -n 50 -m 2
mpiexec -n 1 -machinefile $PBS_NODEFILE ffmpeg -r 72 -y -threads 4 -i ../AnimationData/Spin/%03dspin.png -pix_fmt yuv420p ../AnimationData/Spin/Spin.mp4


echo
echo "Job finished at `date`"
