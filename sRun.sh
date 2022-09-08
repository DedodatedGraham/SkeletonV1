#!/bin/sh
#PBS -l nodes=1:ppn=36
#PBS -o dpost_job.out
#PBS -e dpost_job.err


cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`
echo "processes $num"


mpiexec -n 1 -machinefile $PBS_NODEFILE python3 Main.py -m 2 -p 4

echo
echo "Job finished at `date`"

