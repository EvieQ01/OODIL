import torch.nn as nn
import torch
import numpy as np
from tqdm import trange

def compute_feasibility_traj_discri(expert_traj_pairs_all, expert_trajs, traj_traj_id, discris,\
                                 device='cuda'):
    all_distance = []
    real_criterias = []
    for index in range(len(expert_trajs)): # how many traj in all domains
        frames = []
        all_distance.append([])
        expert_traj_pairs = expert_traj_pairs_all[index]
        discr = discris[traj_traj_id[index]]
        with torch.no_grad():
            # compute discriminator output for experts
            expert_state_action = torch.Tensor(expert_traj_pairs).to(device)
            real = discr(expert_state_action)
            real_criterias.append(real.mean().cpu())
    real_criterias = np.array(real_criterias)
    feasibility = 1 / (1 + np.exp(-real_criterias )) 
    return feasibility
