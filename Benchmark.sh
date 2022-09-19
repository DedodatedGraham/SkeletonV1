#!/bin/sh
#PBS -l nodes=1:ppn=36
#PBS -o bench.out
#PBS -e bench.err

cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`
echo "processes $num"

for i in {1,2,4,8,16,32}
do
    startt=`date +%s`	
    mpiexec -n 1 -machinefile $PBS_NODEFILE python3 Main.py -i 'disk1.dat' -o 'benchsave.dat' -m 1 -p $i
    endt=`date +%s`
    #runtime=$()
    printf "%s\n" "$((endt-startt))" >> bench.txt
done

echo
echo "Job finished at `date`"
