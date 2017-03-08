####  PBS preamble
#!/bin/sh
#PBS -N PBS_test_script
#PBS -M yunfan@umich.edu
#PBS -m abe

#PBS -A eecs542w17_fluxg
#PBS -l qos=flux
#PBS -q fluxg

#PBS -l nodes=1:ppn=1:gpus=1,pmem=16gb
#PBS -l walltime=01:00:00:00
#PBS -j oe
#PBS -V
#PBS -d /scratch/eecs542w17_fluxg/

####  End PBS preamble

if [ -s "$PBS_NODEFILE" ] ; then
    echo "Running on"
    cat $PBS_NODEFILE
fi

if [ -d "$PBS_O_WORKDIR" ] ; then
    cd $PBS_O_WORKDIR
    echo "Running from $PBS_O_WORKDIR"
fi

#  Put your job commands after this line
