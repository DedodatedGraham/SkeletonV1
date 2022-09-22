#!/bin/sh
#PBS -l nodes=1:ppn=128
#PBS -o ../Log/bench.out
#PBS -e ../Log/bench.err

cd $PBS_O_WORKDIR
echo "Starting at $PBS_O_WORKDIR"
echo "node file $PBS_NODEFILE"
num=`cat $PBS_NODEFILE | wc -l`

for i in {1,2,4,8,16,32,64,128}
do
    startt=`date +%s`	
    mpiexec -n 1 -machinefile $PBS_NODEFILE python3 ../Src/Main.py -i 'disk1.dat' -o 'benchsave.dat' -m 1 -p $i
    endt=`date +%s`
    #runtime=$()
    printf "%s\n" "$((endt-startt))" >> ../SkeleData/Output/bench.txt
done

echo
echo "Job finished at `date`"
