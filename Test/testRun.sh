#!/bin/sh
#PBS -l nodes=1:ppn=1
#PBS -o ../Log/test.out
#PBS -e ../Log/test.err


cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"

echo
echo
"Job start at `date`"

mpiexec -n 1 -machinefile $PBS_NODEFILE python3 quadtest.py
echo "Skeleton finished at `date`"


echo
echo "Job finished at `date`"

