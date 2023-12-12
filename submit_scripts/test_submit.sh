#!/bin/sh
### select queue 
#BSUB -q hpc

### name of job, output file and err
#BSUB -J flow_training
#BSUB -o flow_training_%J.out
#BSUB -e flow_training_%J.err

### number of cores
#BSUB -n 1

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 32GB of Memory
#BSUB -R "rusage[mem=32GB]"

### wall time limit - the maximum time the job will run. Lets try 3 hours

#BSUB -W 03:00

##BSUB -u s204090@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 

# end of BSUB options

# load the correct  scipy module and python
module load cuda/11.1

source /zhome/88/9/155925/DL02456_DDPM/venv/bin/activate

python train_flow_cross_validation.py
