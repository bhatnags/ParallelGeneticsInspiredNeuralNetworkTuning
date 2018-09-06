#!/bin/bash

#SBATCH -n 07
#SBATCH -p compute
#SBATCH -t 22:00:00
#SBATCH -J dnnt07
#SBATCH --constraint=LargeMem
#SBATCH --reservation=bhatnags_39


echo "~~~~~~~~~~~~~~~n = 07~~~~~~~~~~~~~~~~~~~~~~~~" >> test.txt
export OMP_NUM_THREADS=1
mpiexec python dnnt.py >> output07dnnt.txt
echo "~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~~~~~~~" >> test.txt

