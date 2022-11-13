import imageio
import matplotlib.pyplot as plt
from tqdm import trange
import argparse
import pdb

import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import Value, Policy, Discriminator

from RL_utils import update_params, select_action
from replay_memory import Memory
from utils import *
from loss import *


import numpy as np



import time
import sys
sys.path.append("../")
sys.path.append("../envs")
import envs


import pickle
import wandb
import matplotlib 
from matplotlib import pyplot as plt
matplotlib.use('Agg')

wandb.init(project="CARLO_feasibility_GAIL2")

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
from dataset_preparation import load_demos

from carlo_utils import make_observation_norm, evaluate, select_init_obs
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', type=str, default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed (default: 1111')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='size of a single batch')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--eval-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train an expert')
parser.add_argument('--hidden-dim', type=int, default=64, metavar='H',
                    help='the size of hidden layers')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='learning rate')
parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                    help='number of iterations of value function optimization iterations per each policy optimization step')
parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                    help='learning rate of value network')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--demo_files', nargs='+', help='the environment used for test')
parser.add_argument('--ratios', nargs='+', type=float, help='the ratio of demos to load')
parser.add_argument('--eval_epochs', type=int, default=10, help='the epochs for evaluation')
parser.add_argument('--save_path', help='the path to save model')
parser.add_argument('--feasibility_model', default=None, help='the path to the feasibility model')
parser.add_argument('--mode', help='the mode of feasibility')
parser.add_argument('--begin-index', type=int, default=10, help='the index of cluster')
parser.add_argument('--init_range', nargs='+', help='the range of init obs.x', default=None)
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--dataset', type=str, default='../demo', help='the source of data root')

args = parser.parse_args()
from logger import *
import json
import copy
logger = CompleteLogger('log/'+ args.env_name + '/2GAIL_feas_'+ args.mode  + \
'_ratio_{}'.format(str([args.ratios])))
args.demo_files = [args.demo_files[0].replace('re_split_simclr_dcn_0_DCN', f"re_split_simclr_dcn_{args.begin_index}_DCN")]
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
wandb.run.name = f"{args.env_name}"+ '_beginindex-{}'.format(args.begin_index)+f"_N{str(len(args.demo_files))}" +\
        now
wandb.config.update(args)

json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)

save_path = logger.get_checkpoint_path('seed_{}_gail_model_begin_index_{}'.format(args.seed, args.begin_index))

demos = []
env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print(f"=> num_actions: {num_actions}", f" \t=> num_states: {num_inputs}")



env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

###### load demos #######################
print('=> run cluster gail')
loaded_demos = load_demos(args.demo_files, args.ratios, num_inputs=num_inputs, args=args)
if len(loaded_demos[0]) == 0:
    exit(0)
expert_pairs, all_trajs = loaded_demos[0:2]
expert_traj = np.concatenate(expert_pairs, axis=0)

policy_net = Policy(num_inputs, num_actions, args.hidden_dim)#.to(device)
value_net = Value(num_inputs, args.hidden_dim).to(device)
discriminator = Discriminator(num_inputs + num_inputs, args.hidden_dim).to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
## continue
if args.continue_train:
    model_dict = torch.load(save_path)
    policy_net.load_state_dict(model_dict['policy'])
    value_net.load_state_dict(model_dict['value'])
    discriminator.load_state_dict(model_dict['disc'])
    save_path = save_path.replace('.pth', '_continue.pth')
    

disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)



def expert_reward(states, actions):
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    with torch.no_grad():
        state_action = torch.Tensor(np.concatenate([states, actions], 1)).to(device)
        return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()



all_idx = np.arange(0, expert_traj.shape[0])
p_idx = np.random.permutation(expert_traj.shape[0])
expert_traj = expert_traj[p_idx, :]

best_reward = -1000000

#### norm ############## (s, s') 
## expert_traj shape as (50000 * 14)
expert_traj = make_observation_norm(expert_traj=expert_traj, env=env, num_inputs=num_inputs)

for i_episode in trange(args.num_epochs):
    env.seed(int(time.time()))
    memory = Memory()

    num_steps = 0
    num_episodes = 0
    
    reward_batch = []
    states = []
    actions = []
    next_states = []
    mem_actions = []
    mem_mask = []
    mem_next = []
    # evaluate(i_episode, best_reward, log_file)

    time_sample = time.time()
    while num_steps < args.batch_size:

        env.reset()
        state = env.reset_with_obs(select_init_obs(all_trajs), after_norm=False)
        # state = env.reset()
   

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            action = select_action(state, policy_net=policy_net)
            action = action.data[0].numpy()
            states.append(np.array([state]))
            actions.append(np.array([action]))
            next_state, true_reward, done, _ = env.step(action)
            next_states.append(np.array([next_state]))
            reward_sum += true_reward

            mask = 1
            if done:
                mask = 0

            mem_mask.append(mask)
            mem_next.append(next_state)
            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1

        reward_batch.append(reward_sum)
    time_sample = time.time() - time_sample
    if i_episode % args.eval_interval == 0:

        print('=> save at: ', save_path)
        best_reward = evaluate(env=env, policy_net=policy_net, value_net=value_net, discriminator=discriminator, episode=i_episode, best_reward=best_reward,\
                            save_path=save_path, args=args, all_trajs=all_trajs)

    rewards = expert_reward(states, next_states)
    
    for idx in range(len(states)):
        memory.push(states[idx][0], actions[idx], mem_mask[idx], mem_next[idx], \
                    rewards[idx][0])
    batch = memory.sample()
    time_update_param = time.time()
    update_params(batch, policy_net, value_net, value_optimizer, value_criterion, args, device='cpu')
    time_update_param = time.time() - time_update_param

    ### update discriminator ###
    time_train_disc = time.time()
    next_states = torch.from_numpy(np.concatenate(next_states))
    states = torch.from_numpy(np.concatenate(states))

    labeled_num = min(expert_traj.shape[0], num_steps)

    idx = np.random.choice(all_idx, labeled_num)

    expert_state_action = expert_traj[idx, :]
    expert_state_action = torch.Tensor(expert_state_action).to(device)
    real = discriminator(expert_state_action)    

    state_action = torch.cat((states, next_states), 1).to(device)
    fake = discriminator(state_action)

    disc_optimizer.zero_grad()
    disc_loss = disc_criterion(fake, torch.ones(fake.size(0), 1).to(device)) + \
                disc_criterion(real, torch.zeros(real.size(0), 1).to(device))

    disc_loss.backward()
    disc_optimizer.step()
    time_train_disc = time.time() - time_train_disc
    ############################
    if i_episode % args.log_interval == 0:
        print('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))
        # writer.add_scalar('gail/loss{}', disc_loss.item(), i_episode)
        wandb.log({
            'output/Mean train reward: ': np.mean(reward_batch),
            'output/Max train reward: ': max(reward_batch)
        })

    wandb.log({
        'debug/discriminator real: ': torch.mean(real),
        'debug/disctiminator fake: ': torch.mean(fake),
        'debug/disctiminator loss: ': disc_loss,
        'debug/r(s, s_hat)': np.mean(rewards),
        'Timing/rollout': time_sample,
        'Timing/RL': time_update_param,
        'Timing/IRL:train discri': time_train_disc,
    })
