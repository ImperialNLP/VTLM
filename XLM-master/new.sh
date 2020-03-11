#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output rapor.out.%j
bash bla.sh
