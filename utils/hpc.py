import os
import re

def combine_results(output_dir:str):
    """This function combines the results of the SLURM array jobs into a single file. It checks if all jobs have finished successfully and then combines the results into a single file."""

    if int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)) == int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1)) - 1:
        # with while loop, check if all jobs have finished successfully
        still_running = True
        num_running_jobs = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
        pattern = re.compile(r"results_\d+_finished\.csv$")
        while still_running:
            # number of finished jobs
            num_finished_jobs = len([f for f in os.listdir(output_dir) if pattern.search(f)])
            if num_finished_jobs >= num_running_jobs:
                still_running = False
        
        # combine results
        for i in range(num_running_jobs):
            if i != 0:
                with open(os.path.join(output_dir, f'results_{i}_finished.csv'), 'r') as f:
                    data = f.readlines()
                with open(os.path.join(output_dir, 'results.csv'), 'a') as f:
                    f.writelines(data[1:])
                os.remove(os.path.join(output_dir, f'results_{i}_finished.csv'))
            else:
                os.rename(os.path.join(output_dir, f'results_{i}_finished.csv'), os.path.join(output_dir, 'results.csv'))
    print('Combined results')


def divide_list_slurm_array(lst:list) -> list:
    """Used to divide a list into sublists for use with SLURM array jobs. 
    It outputs the sublist that corresponds to the current job.

    Args:
        lst (list): List to divide

    Returns:
        list: A sublist of the input list
    """

    slurm_array_task_count = int(os.environ.get('SLURM_ARRAY_TASK_COUNT', 1))
    slurm_array_job_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    division_size = len(lst) // slurm_array_task_count
    divisions = [lst[i * division_size:(i + 1) * division_size] for i in range(slurm_array_task_count - 1)]
    divisions.append(lst[(slurm_array_task_count - 1) * division_size:])
    return divisions[slurm_array_job_id]