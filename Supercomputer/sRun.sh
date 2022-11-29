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
mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/Main.py -m 1 -p 128 -i 'infc_50.dat' -o 'infc_50_save.dat'
echo
echo "Skeleton finished at `date`"


echo

