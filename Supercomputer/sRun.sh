#!/bin/sh
#PBS -l nodes=1:ppn=128
#PBS -o ../Log/sRun_job.out
#PBS -e ../Log/sRun_job.err


cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`
echo "processes $num"


mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/Main.py -m 1 -p 120
echo "Skeleton finished at `date`"

mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/3DSpin.py -n 30
echo "Plot finished at `date`"


echo
echo "Job finished at `date`"

