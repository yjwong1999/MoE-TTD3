import numpy as np
import pandas as pd
import openpyxl, os, copy, shutil
import argparse

# get argument
parser = argparse.ArgumentParser()

parser.add_argument('--user-num', type = int, required = False, default=2, help="how many users")
parser.add_argument('--attacker-num', type = int, required = False, default=1, help="how many attacker/eavesdropper")
parser.add_argument('--task-num', type = int, required = False, default=50, help="how many tasks")

args = parser.parse_args()

# get tasks configs
USER_NUM = args.user_num
ATTACKER_NUM = args.attacker_num
TASK_NUM = args.task_num

SYSTEM_BORDER = [(-25,25), (0, 50)]


def mobile_entity(worksheet, num):
    '''
    Purpose:
    To create config for mobile entity except uav

    args:
        1. worksheet: worksheet to store info about this entity
        2. num      : number of such entity
    '''

    worksheet.append(['index','x','y','z','delta_d','direction_fai_coef'])
    for idx in range(num):
        # init xyz location
        x = np.random.uniform(SYSTEM_BORDER[0][0],SYSTEM_BORDER[0][1])
        y = np.random.uniform(SYSTEM_BORDER[1][0],SYSTEM_BORDER[1][1])
        z = 0.0001
        # trajectory setting (speed and angle)
        delta_d = np.random.uniform(0.05, 0.2)
        direction_fai_coef = np.random.uniform(-1.0, 1.0)
        # add to user sheet
        row = [idx, x, y, z, delta_d, direction_fai_coef]
        worksheet.append(row)
    return worksheet

def main():
    '''
    Purpose:
    loop and create a excel file as a config for all task setup
    '''
    
    # make directory
    os.mkdir('data/init')

    # set seed for reproducibility
    np.random.seed(2702)

    # loop for all task
    for task_idx in range(TASK_NUM):

        # create workbook for this task setup
        workbook = openpyxl.Workbook()

        # Create sheets for this workbook
        worksheets = [workbook.create_sheet(sheet_name) for sheet_name in ['user', 'attacker', 'RIS', 'RIS_norm_vec', 'UAV']]
        
        # Write data to user sheet
        worksheets[0] = mobile_entity(worksheets[0], USER_NUM)

        # Write data to user sheet
        worksheets[1] = mobile_entity(worksheets[1], ATTACKER_NUM)

        # Write data to RIS sheet
        worksheets[2].append(['index','x','y','z'])
        worksheets[2].append([0, 0.0001, 50, 12.5])

        # Write data to RIS_norm_vec sheet
        worksheets[3].append(['index','x','y','z'])
        worksheets[3].append([0, 0, -1, 0])

        # Write data to RIS_norm_vec sheet
        worksheets[4].append(['index','x','y','z'])
        worksheets[4].append([0, 0.0001, 25, 50])
        
        # Save the workbook
        workbook.save(f"data/init/{task_idx + 1}_init_location.xlsx")

    print('Finished creating configs!')


if __name__ == "__main__":
    # make sure you do not overwrite previous configs (in case you still need them)
    if os.path.isdir('data/init'):
        # prompt user
        overwrite = input('You have previously created configs for the task! \n Do you wish to overwrite (Y/n): ')
    
        # if overwrite
        if overwrite.lower() == 'y':
            shutil.rmtree('data/init')
            main()
        else:
            print('\nNo overwrite, stopping now')  
    else:
        main()
