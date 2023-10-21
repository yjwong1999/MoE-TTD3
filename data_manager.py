import numpy as np
import scipy.io
import pandas as pd
import os, shutil
import time, csv

class DataManager(object):
    """
    class to read and store simulation results
    before use, please create a direction under current file path './data'
    and must have a file 'init_location.xlsx' which contain the position of each entities
    """
    def __init__(self, store_list = ['beamforming_matrix', 'reflecting_coefficient', 'UAV_state', 'user_capacity'],\
                 file_path = './data', project_name = None, store_path = './data/storage'):
        # 1 init location data
        self.store_list = store_list
        self.init_data_file = file_path + '/init_location.xlsx'
        if project_name is None:
            self.time_stemp = time.strftime('/%Y-%m-%d %H_%M_%S',time.localtime(time.time()))
            self.store_path = store_path + self.time_stemp 
        else:
            dir_name = store_path + '/' + project_name
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
            self.store_path = dir_name

        os.makedirs(self.store_path) 
        # self.writer = pd.ExcelWriter(self.store_path + '/simulation_result.xlsx', engine='openpyxl')  # pylint: disable=abstract-class-instantiated 
        self.simulation_result_dic = {}
        self.init_format()

    def save_file(self, episode_cnt = 10):
        # record step counts per episode
        with open(self.store_path + "/step_num_per_episode.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([len(list(self.simulation_result_dic.values())[0])])

        # when ended, auto save to .mat file
        scipy.io.savemat(self.store_path + '/simulation_result_ep_' + str(episode_cnt) + '.mat', {'result_' + str(episode_cnt):self.simulation_result_dic})
        self.simulation_result_dic = {}
        self.init_format()

    def save_meta_data(self, meta_dic):
        """
        save system and agent information
        """
        scipy.io.savemat(self.store_path + '/meta_data.mat', {'meta_data': meta_dic})
        
    def init_format(self):
        """
        used only one time in env.py
        """
        for store_item in self.store_list:
            self.simulation_result_dic.update({store_item:[]})

    def read_init_location(self, entity_type = 'user', index = 0):
        temp = []
        if entity_type in ['user', 'attacker', 'RIS', 'RIS_norm_vec', 'UAV']:
            # xyz
            temp.append(pd.read_excel(self.init_data_file, sheet_name=entity_type)['x'][index])
            temp.append(pd.read_excel(self.init_data_file, sheet_name=entity_type)['y'][index])
            temp.append(pd.read_excel(self.init_data_file, sheet_name=entity_type)['z'][index])
            return np.array(temp)
        else:
            assert False
            
    def read_movement(self, entity_type = 'user', index = 0):
        temp = []
        if entity_type in ['user', 'attacker']:
            # movement config
            delta_d = pd.read_excel(self.init_data_file, sheet_name=entity_type)['delta_d'][index]
            direction_fai_coef = pd.read_excel(self.init_data_file, sheet_name=entity_type)['direction_fai_coef'][index]
            return delta_d, direction_fai_coef
        else:
            assert False
    
    def store_data(self, row_data, value_name):
        """
        docstring
        """
        self.simulation_result_dic[value_name].append(row_data)
