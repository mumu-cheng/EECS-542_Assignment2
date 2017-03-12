####  PBS preamble
#!/bin/sh
#PBS -N train_fcn16_fluxg_542w17
#PBS -M yunfan@umich.edu
#PBS -m abe

#PBS -A eecs542w17_fluxg
#PBS -l qos=flux
#PBS -q fluxg

#PBS -l nodes=1:ppn=1:gpus=1,pmem=16gb
#PBS -l walltime=01:12:00:00
#PBS -j oe
#PBS -V
#PBS -d /home/yunfan

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

cd ./EECS-542_Assignment2/Code && th train_16.lua
