import math

import numpy as np

import torch


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (
        2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

def process_expert_traj(expert_traj_raw):
    expert_traj = []
    for i in range(len(expert_traj_raw)):
        for j in range(len(expert_traj_raw[i])):
            expert_traj.append(expert_traj_raw[i][j])
    expert_traj = np.stack(expert_traj)
    #print('here',  expert_traj.shape)
    return expert_traj

def generate_pairs(expert_traj_raw, state_indices, step_size=1):
    '''
    generate state pairs (s, s_t)
    note that s_t can be multi-step future (controlled by max_step)
    '''
    pairs = []

    states = expert_traj_raw['obs']
    next_states = expert_traj_raw['next_obs']
    for i in range(len(states)):
        if state_indices is not None:
            state_traj = np.array(states[i])[:, state_indices]
            next_state_traj = np.array(next_states[i])[:, state_indices]
        else:
            state_traj = np.array(states[i])
            next_state_traj = np.array(next_states[i])
        if len(state_traj) == 0:
            continue
        next_state_traj = next_state_traj[step_size-1:]
        pairs.append(np.concatenate([state_traj[0:next_state_traj.shape[0]], next_state_traj], axis=1))
    pairs = np.concatenate(pairs, axis=0)
    return pairs

def generate_tuples(expert_traj_raw, state_dim):
    '''
    generate transition tuples (s, s', a) for training
    '''
    expert_traj = []
    for i in range(len(expert_traj_raw)):
        for j in range(len(expert_traj_raw[i])):
            if j < len(expert_traj_raw[i])-1:
                state_action = expert_traj_raw[i][j]
                next_state = expert_traj_raw[i][j+1][:state_dim]
                transitions = np.concatenate([state_action[:state_dim], next_state, state_action[state_dim:]], axis=-1)
                expert_traj.append(transitions)
    expert_traj = np.stack(expert_traj)
    return expert_traj

def adjust_lr(optimizer, scale):
    print('=========adjust learning rate================')
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] / scale


def normalize_expert_traj(running_state, expert_traj, state_dim):
    '''
    normalize the demonstration data by the state normalizer
    '''
    traj = []
    for i in range(len(expert_traj)):
        state = expert_traj[i, :state_dim]
        rest = expert_traj[i, state_dim:]
        state = running_state(state, update=False)
        tuple = np.concatenate([state, rest], axis=-1)
        traj.append(tuple)
    traj = np.stack(traj)
    return traj

def normalize_states(running_state, state_pairs, state_dim):
    '''
    normalize the state pairs/tuples by state normalizer
    '''
    traj = []
    for i in range(len(state_pairs)):
        state = state_pairs[i, :state_dim]
        next_state = state_pairs[i, state_dim:state_dim*2]
        rest = state_pairs[i, state_dim*2:]
        state = running_state(state, update=False)
        next_state = running_state(next_state, update=False)
        tuple = np.concatenate([state, next_state, rest], axis=-1)
        traj.append(tuple)
    traj = np.stack(traj)
    return traj


