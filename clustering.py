# debug field
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import argparse

# get argument from user
parser = argparse.ArgumentParser()
parser.add_argument('--drl', type = str, required = True, default='td3', help="which drl algo would you like to choose ['ddpg', 'td3']")
parser.add_argument('--reward', type = str, required = True, default='see', help="which reward would you like to implement ['ssr', 'see']")
parser.add_argument('--ep-num', type = int, required = False, default=300, help="how many episodes do you want to train your DRL")
parser.add_argument('--task-num', type = int, required = True, default=None, help="how many tasks")

# get the arguments
args = parser.parse_args()
DRL_ALGO = args.drl
REWARD_DESIGN = args.reward
EPISODE_NUM = args.ep_num
TASK_NUM = args.task_num

# validate the argument
assert DRL_ALGO in ['ddpg', 'td3'], "drl must be ['ddpg', 'td3']"
assert REWARD_DESIGN in ['ssr', 'see'], "reward must be ['ssr', 'see']"

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


###########################################
# Get weights of all models
###########################################
all_params = [] # parameters of all task-based DRL
for i in range(TASK_NUM):
    project_name = str(i + 1)

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

    # get model parameters
    params = []
    for key in agent_1.actor._modules.keys():
        params += [p.view(-1) for p in agent_1.actor._modules[key].parameters()]
    for key in agent_2.actor._modules.keys():
        params += [p.view(-1) for p in agent_2.actor._modules[key].parameters()]

    params = torch.cat(params)

    all_params.append(params)


###########################################
# cosine similarity matrix
###########################################
cos = torch.nn.CosineSimilarity(dim=0)
sim_matrix = np.zeros((TASK_NUM, TASK_NUM))
for i in range(TASK_NUM):
    for j in range(TASK_NUM):
        sim = cos(all_params[i], all_params[j])
        #print(i+1, j+1, sim)
        sim_matrix[i][j] = sim
print(sim_matrix)


###########################################
# read results
###########################################
# read json file
import json
with open('result.json', "r") as read_file:
    results = json.load(read_file)
results


###########################################
# clustering
###########################################
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


X = []
for i in range(len(sim_matrix)):
    sim_with_other_model = sim_matrix[i]
    result_of_this_model = results[str(i+1)]
    # print(list(sim_with_other_model))

    feature = [result_of_this_model['reward'],
              result_of_this_model['ave_ssr']]
    feature += list(sim_with_other_model)
    #print(feature)

    X.append(feature)

# Create a KMeans object with 3 clusters
clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X)
clustering.labels_

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(clustering, truncate_mode="level", p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
