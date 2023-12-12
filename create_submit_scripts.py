import os



def createSubmitScripts():
    # loop over models in the folder
    for idx,filename in enumerate(sorted(os.listdir('untrained_models'))):
                        

            script_template = '''
            #!/bin/sh
            ### select queue 
            #BSUB -q hpc

            ### name of job, output file and err
            #BSUB -J flow_training_{idx}
            #BSUB -o flow_training_{idx}_%J.out
            #BSUB -e flow_training_{idx}_%J.err

            ### number of cores
            #BSUB -n 1

            ### -- specify that the cores must be on the same host -- 
            #BSUB -R "span[hosts=1]"
            ### -- specify that we need 32GB of Memory
            #BSUB -R "rusage[mem=32GB]"

            ### wall time limit - the maximum time the job will run. Lets try 6.5 hours

            #BSUB -W 06:30

            ##BSUB -u s204090@dtu.dk
            ### -- send notification at start -- 
            #BSUB -B 
            ### -- send notification at completion -- 
            #BSUB -N 

            # end of BSUB options

            # load the correct  scipy module and python
            module load cuda/11.1

            source /zhome/88/9/155925/DL02456_DDPM/venv/bin/activate

            python train_flow_cross_validation.py --model_name {model_name}
            '''

            script_content = script_template.format(idx=idx, model_name=filename)
            with open(f'submit_scripts/job_{idx}.sh', 'w') as fp:
                fp.write(script_content)

if __name__ == "__main__":
    createSubmitScripts()