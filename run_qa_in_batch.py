import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
    
    # Path specific arguments
parser.add_argument('--batch_json_path', required=True, type=str, help='Path to json containing batch job.')
parser.add_argument('--use_this_gpu', required=False, type=str, help='Allows the use of only 1 gpu. Specify the id of the gpu you would like to use.', default='None')
parser.add_argument('-e','--evaluate_only',action='store_true')


arg_dict = vars(parser.parse_args())  

with open(arg_dict['batch_json_path'], 'r') as job:
    batch_data = json.load(job)['models']

python_file = 'bert_qa_finetune.py' if not arg_dict['evaluate_only'] else 'bert_qa_evaluate.py'

for single_job in batch_data:
    if arg_dict['use_this_gpu'] != 'None':
        python_run_command = 'CUDA_VISIBLE_DEVICES=\"' + arg_dict['use_this_gpu'] + "\" python " + python_file + " "
    else: 
        python_run_command = "python " + python_file + " "
    for key, value in single_job.items():
        python_run_command += f'--{key} {value} '
    
    subprocess.call(python_run_command, shell=True)