#!/bin/sh
#PBS -l nodes=1:ppn=128
#PBS -o ../Log/3D.out
#PBS -e ../Log/3D.err

cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`

mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/3DSpin.py -n 100


echo
echo "Job finished at `date`"
