#!/bin/sh
#PBS -l nodes=1:ppn=128
#PBS -o ../Log/sRun_job.out
#PBS -e ../Log/sRun_job.err


cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`
echo "processes $num"

echo "Skeleton started at `date`"
echo
mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/Main.py -m 1 -p 128 -i 'infc_0.dat' -o 'infc_0_save.dat'
echo
echo "Skeleton finished at `date`"
mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/3DSpin.py -i 'Output/infc_50_savePrePurge.dat' -n 30 -m 0
mpiexec -n 1 -machinefile $PBS_NODEFILE ffmpeg -r 72 -y -threads 4 -i ../AnimationData/Spin/%03dspin.png -pix_fmt yuv420p ../AnimationData/Spin/Spin.mp4
echo
echo "Plot finished at `date`"

echo

