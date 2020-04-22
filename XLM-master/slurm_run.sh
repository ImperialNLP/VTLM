#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:3
#SBATCH --output=./%j.stdout
#SBATCH --error=./%j.stderr
#SBATCH --job-name="xlm"
#SBATCH --open-mode=append

LOG_STDOUT="/data/menekse/$SLURM_JOB_ID.stdout"
LOG_STDERR="/data/menekse/$SLURM_JOB_ID.stderr"

# Start (or restart) experiment
date >> $LOG_STDOUT
which python >> $LOG_STDOUT
echo "---Beginning program ---" >> $LOG_STDOUT
echo "Exp name     : xlm" >> $LOG_STDOUT
echo "SLURM Job ID : $SLURM_JOB_ID" >> $LOG_STDOUT
echo "SBATCH script: bla.sh" >> $LOG_STDOUT

srun --label bash bla.sh 10100 &

wait $!
