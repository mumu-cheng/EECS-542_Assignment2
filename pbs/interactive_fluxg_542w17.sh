#!/bin/bash
qsub \
-I \
-S /bin/sh \
-N caffe_make \
-l nodes=1:ppn=1:gpus=1 \
-l pmem=16gb \
-q fluxg \
-A eecs542w17_fluxg \
-l qos=flux \
-M yunfan@umich.edu \
-m abe \
-l walltime=1:00:00:00 \
-j eo \
-V \
-d "/scratch/eecs542w17_fluxg/yunfan/"
