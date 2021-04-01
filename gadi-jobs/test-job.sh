#!/bin/bash
#PBS -P hj72
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l walltime=0:05:00,mem=8GB
#PBS -l wd


module load pytorch/1.4.0a0 python3/3.7.4 cuda/10.1

python3 -m models.train testconfig.ini > testrun.log

