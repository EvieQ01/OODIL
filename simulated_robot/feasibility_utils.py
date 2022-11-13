import torch.nn as nn
import torch
import numpy as np
from tqdm import trange

from gail_airl_ppo.buffer import ConcatStateBuffers
def compute_feasibility_traj_discri(concat_buffer:ConcatStateBuffers, discris,\
                                device='cuda'):
    '''
    calculate feasibility for each trajectory.
    e.g.
        concatbuffer.domain_traj_count = [100, 100, 100] 3 domains
        concatbuffer.traj_len = [999, 999, 999,...] n_trajs in all
    '''
    real_criterias = []
    begin_traj_id = 0
    for index in range(len(concat_buffer.domain_traj_count)):# domain count
        # for each domain
        curr_traj_id = 0 # always plus begin_traj_id befor use.
        discr = discris[index]
        with torch.no_grad():
            while curr_traj_id < concat_buffer.domain_traj_count[index]:
                # for each traj
                expert_state = concat_buffer.get_all_pairs_for_traj(curr_traj_id + begin_traj_id)
                expert_next_state = concat_buffer.get_all_next_pairs_for_traj(curr_traj_id + begin_traj_id)
                curr_traj_id += 1
                # compute discriminator output for experts
                expert_state = torch.tensor(expert_state).to(device)
                expert_next_state = torch.tensor(expert_next_state).to(device)
                real = discr(expert_state, expert_next_state)
                real_criterias.append(real.mean().cpu())
            begin_traj_id += concat_buffer.domain_traj_count[index]
    
    # shape: n_trajs
    real_criterias = np.array(real_criterias)
    # real label is '1'# [Note]: This is different from mujoco. and Carlo
    sigmoid = 1 - 1 / (1 + np.exp(-real_criterias)) 
    exp = np.exp(sigmoid / 0.1) 
    feasibility = (exp - exp.min()) / (exp.max() - exp.min())
    
    # evaluate:
    begin_idx = 0
    for i in range(len(concat_buffer.domain_traj_count)):
        sum_feas = feasibility[begin_idx : concat_buffer.domain_traj_count[i] + begin_idx].sum()
        # print(f'=> mean feasibility for cluster {i}: {mean_feas}')
        print(f'=> sum feasibility for cluster {i}: {sum_feas}')
        begin_idx += concat_buffer.domain_traj_count[i]
    return feasibility
