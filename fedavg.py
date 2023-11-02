# debug field
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse


# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--drl', type = str, required = True, default='td3', help="which drl algo would you like to choose ['ddpg', 'td3']")
parser.add_argument('--reward', type = str, required = True, default='see', help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--ep-num', type = int, required = False, default=300, help="how many episodes do you want to train your DRL")
parser.add_argument('--seeds', type = int, required = False, default=None,  nargs='+', help="what seed(s) would you like to use for DRL 1 and 2, please provide in one or two int")
parser.add_argument('--task-num', type = int, required = True, default=None, help="how many tasks")
parser.add_argument('--task-idx', type = int, required = False, default=None,  nargs='+', help="task index")
parser.add_argument('--split-idx', type = str, required = True, default='1', help="which split now")

# get the arguments
args = parser.parse_args()
DRL_ALGO = args.drl
REWARD_DESIGN = args.reward
EPISODE_NUM = args.ep_num
SEEDS = args.seeds
TASK_NUM = args.task_num
TASK_IDX = args.task_idx
SPLIT_IDX = args.split_idx

# validate the argument
assert DRL_ALGO in ['ddpg', 'td3'], "drl must be ['ddpg', 'td3']"
assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"
if SEEDS is not None:
    assert len(SEEDS) in [1, 2] and isinstance(SEEDS[0], int) and isinstance(SEEDS[-1], int), "seeds must be a list of 1 or 2 integer"
if TASK_IDX is not None:
    TASK_NUM = len(TASK_IDX)
else:
    TASK_IDX = [i + 1 for i in range(TASK_NUM)]

# get DRL_ALGO
if DRL_ALGO == 'td3':
    from td3 import Agent
elif DRL_ALGO == 'ddpg':
    from ddpg import Agent

from env import MiniSystem
import numpy as np
import math
import time
import torch
import shutil

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr, pearsonr, rankdata

###########################################
# Get weights of all models
###########################################
all_agent_1_params = []
all_agent_2_params = []

for i in range(TASK_NUM):
    i = TASK_IDX[i]
    project_name = str(i)

    # 1 init system model
    step_num = 20

    system = MiniSystem(
        user_num=2,
        RIS_ant_num=4,
        UAV_ant_num=4,
        if_dir_link=1,
        if_with_RIS=True,
        if_move_users=True,
        if_movements=True,
        if_UAV_pos_state = True,
        reward_design = REWARD_DESIGN,
        project_name = project_name,
        train=False
        )

    if_Theta_fixed = False
    if_G_fixed = False
    if_BS = False
    if_robust = True


    # 2 init RL Agent
    if SEEDS is not None:
        torch.manual_seed(SEEDS[0]) # 1
        torch.cuda.manual_seed_all(SEEDS[0]) # 1
    agent_1_param_dic = {}
    agent_1_param_dic["alpha"] = 0.0001
    agent_1_param_dic["beta"] = 0.001
    agent_1_param_dic["input_dims"] = system.get_system_state_dim()
    agent_1_param_dic["tau"] = 0.001
    agent_1_param_dic["batch_size"] = 64
    agent_1_param_dic["n_actions"] = system.get_system_action_dim() - 2
    agent_1_param_dic["action_noise_factor"] = 0.1
    agent_1_param_dic["memory_max_size"] = int(5/5 * step_num) #/2
    agent_1_param_dic["agent_name"] = "G_and_Phi"
    agent_1_param_dic["layer1_size"] = 800
    agent_1_param_dic["layer2_size"] = 600
    agent_1_param_dic["layer3_size"] = 512
    agent_1_param_dic["layer4_size"] = 256
    
    if SEEDS is not None:
        torch.manual_seed(SEEDS[-1]) # 2
        torch.cuda.manual_seed_all(SEEDS[-1]) # 2
    agent_2_param_dic = {}
    agent_2_param_dic["alpha"] = 0.0001
    agent_2_param_dic["beta"] = 0.001
    agent_2_param_dic["input_dims"] = 3
    agent_2_param_dic["tau"] = 0.001
    agent_2_param_dic["batch_size"] = 64
    agent_2_param_dic["n_actions"] = 2
    agent_2_param_dic["action_noise_factor"] = 0.5
    agent_2_param_dic["memory_max_size"] = int(5/5 * step_num) #/2
    agent_2_param_dic["agent_name"] = "UAV"
    agent_2_param_dic["layer1_size"] = 400
    agent_2_param_dic["layer2_size"] = 300
    agent_2_param_dic["layer3_size"] = 256
    agent_2_param_dic["layer4_size"] = 128

    agent_1 = Agent(
        alpha       = agent_1_param_dic["alpha"],
        beta        = agent_1_param_dic["beta"],
        input_dims  = [agent_1_param_dic["input_dims"]],
        tau         = agent_1_param_dic["tau"],
        env         = system,
        batch_size  = agent_1_param_dic["batch_size"],
        layer1_size=agent_1_param_dic["layer1_size"],
        layer2_size=agent_1_param_dic["layer2_size"],
        layer3_size=agent_1_param_dic["layer3_size"],
        layer4_size=agent_1_param_dic["layer4_size"],
        n_actions   = agent_1_param_dic["n_actions"],
        max_size = agent_1_param_dic["memory_max_size"],
        agent_name= agent_1_param_dic["agent_name"]
        )

    agent_2 = Agent(
        alpha       = agent_2_param_dic["alpha"],
        beta        = agent_2_param_dic["beta"],
        input_dims  = [agent_2_param_dic["input_dims"]],
        tau         = agent_2_param_dic["tau"],
        env         = system,
        batch_size  = agent_2_param_dic["batch_size"],
        layer1_size=agent_2_param_dic["layer1_size"],
        layer2_size=agent_2_param_dic["layer2_size"],
        layer3_size=agent_2_param_dic["layer3_size"],
        layer4_size=agent_2_param_dic["layer4_size"],
        n_actions   = agent_2_param_dic["n_actions"],
        max_size = agent_2_param_dic["memory_max_size"],
        agent_name= agent_2_param_dic["agent_name"]
        )


    # 3 load task-specific weight
    if DRL_ALGO == 'td3':
        agent_1.load_models(
            load_file_actor = system.data_manager.store_path.replace('test', 'train') + '/Actor_G_and_Phi_TD3',
            load_file_critic_1 = system.data_manager.store_path.replace('test', 'train') + '/Critic_1_G_and_Phi_TD3',
            load_file_critic_2 = system.data_manager.store_path.replace('test', 'train') + '/Critic_2_G_and_Phi_TD3'
            )
        agent_2.load_models(
            load_file_actor = system.data_manager.store_path.replace('test', 'train') + '/Actor_UAV_TD3',
            load_file_critic_1 = system.data_manager.store_path.replace('test', 'train') + '/Critic_1_UAV_TD3',
            load_file_critic_2 = system.data_manager.store_path.replace('test', 'train') + '/Critic_2_UAV_TD3'
            )
    elif DRL_ALGO == 'ddpg':
        agent_1.load_models(
            load_file_actor = system.data_manager.store_path.replace('test', 'train') + '/Actor_G_and_Phi_ddpg',
            load_file_critic = system.data_manager.store_path.replace('test', 'train') + '/Critic_G_and_Phi_ddpg'
            )
        agent_2.load_models(
            load_file_actor = system.data_manager.store_path.replace('test', 'train') + '/Actor_UAV_ddpg',
            load_file_critic = system.data_manager.store_path.replace('test', 'train') + '/Critic_UAV_ddpg'
            )

    # 4 get model parameters

    # 4.1 extract params
    agent_1_params = [
                        dict(agent_1.actor.named_parameters()),
                        dict(agent_1.critic_1.named_parameters()),
                        dict(agent_1.critic_2.named_parameters()),
                        dict(agent_1.target_actor.named_parameters()),
                        dict(agent_1.target_critic_1.named_parameters()),
                        dict(agent_1.target_critic_2.named_parameters()),
                    ]

    agent_2_params = [
                        dict(agent_2.actor.named_parameters()),
                        dict(agent_2.critic_1.named_parameters()),
                        dict(agent_2.critic_2.named_parameters()),
                        dict(agent_2.target_actor.named_parameters()),
                        dict(agent_2.target_critic_1.named_parameters()),
                        dict(agent_2.target_critic_2.named_parameters()),
                    ]
    
    # 4.2 append
    all_agent_1_params.append(agent_1_params)
    all_agent_2_params.append(agent_2_params)
hardcode_index = i

# 5 perform FL
print("\nFEDERATED LEARNING\n")

# 5.1 get all netwworks in agent 1 and 2
agent_1_nets = [
            agent_1.actor,
            agent_1.critic_1,
            agent_1.critic_2,
            agent_1.target_actor,
            agent_1.target_critic_1,
            agent_1.target_critic_2,
            ]
agent_2_nets = [
            agent_2.actor,
            agent_2.critic_1,
            agent_2.critic_2,
            agent_2.target_actor,
            agent_2.target_critic_1,
            agent_2.target_critic_2,
            ]


# 5.2 perform FL on agent 1
for i in range(len(agent_1_params)): # i is the 6 networks in agent
    params = agent_1_params[i]

    new_param = None
    for param_name in params:
        alist = [ all_agent_1_params[j][i][param_name][None] for j in range(TASK_NUM) ] # j is task number
        new_param = torch.stack(alist, dim=0).sum(dim=0).sum(dim=0)
        new_param /= TASK_NUM
        params[param_name] = new_param

    agent_1_nets[i].load_state_dict(params)

# 5.3 perform FL on agent 2
for i in range(len(agent_2_params)):
    params = agent_2_params[i]

    new_param = None
    for param_name in params:
        alist = [ all_agent_2_params[j][i][param_name][None] for j in range(TASK_NUM) ]
        new_param = torch.stack(alist, dim=0).sum(dim=0).sum(dim=0)
        new_param /= TASK_NUM
        params[param_name] = new_param

    agent_2_nets[i].load_state_dict(params)


# 5.4 change checkpoint name
for i in range(len(agent_1_params)):
    agent_1_nets[i].checkpoint_file = agent_1_nets[i].checkpoint_file.replace('test', 'seed').replace(f'{hardcode_index}', f'{SPLIT_IDX}')
    print(agent_1_nets[i].checkpoint_file)
for i in range(len(agent_2_params)):
    agent_2_nets[i].checkpoint_file = agent_2_nets[i].checkpoint_file.replace('test', 'seed').replace(f'{hardcode_index}', f'{SPLIT_IDX}')
    print(agent_2_nets[i].checkpoint_file)


# 5.5 save FedAvg checkpoints
dirname = os.path.dirname(agent_2_nets[i].checkpoint_file)
rootname = os.path.dirname(dirname)
if not os.path.isdir(rootname):
    os.mkdir(rootname)
if not os.path.isdir(dirname):
    os.mkdir(dirname)

agent_1.save_models()
agent_2.save_models()
