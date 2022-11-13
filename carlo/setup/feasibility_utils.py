import torch.nn as nn
import torch
import numpy as np
from tqdm import trange
import pdb
def compute_feasibility_traj_discri(expert_traj_pairs_all, expert_trajs, traj_traj_id, models, discris,\
                                f_env, init_obs, count_of_traj, args, device='cuda', norm=False, return_dsa_info=False):
    disc_criterion = nn.BCEWithLogitsLoss()
    all_distance = []
    fake_criterias = []
    real_criterias = []
    disc_loss_criterias = []
    for index in range(len(expert_trajs)): # how many traj in all domains
        frames = []
        # frames = np.empty(())
        all_distance.append([])
        expert_traj = expert_trajs[index]
        
        expert_traj_pairs = expert_traj_pairs_all[index]
        model = models[traj_traj_id[index]]
        discr = discris[traj_traj_id[index]]
        with torch.no_grad():

            # compute discriminator output for experts
            expert_state_action = torch.Tensor(expert_traj_pairs).to(device)
            real = discr(expert_state_action)
            real_criterias.append(real.mean().cpu())
    real_criterias = np.array(real_criterias)
    
    feasibility = 1 / (1 + np.exp(-real_criterias))
    return feasibility
