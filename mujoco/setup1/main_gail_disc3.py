import imageio
import matplotlib.pyplot as plt
from tqdm import trange
import argparse
from itertools import count
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
from models.old_models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from utils import *
from loss import *

import numpy as np

import time
import sys
sys.path.append('../all_envs')

import walker
import halfcheetah
import hopper

import pickle
import wandb
from feasibility_utils import * 
from demo_utils import load_demos
from RL_utils import select_action, update_params
wandb.init(project="DifferentDynamics_disc_feas")

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = 'cpu'
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
parser.add_argument('--eval-interval', type=int, default=1, metavar='N',
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
parser.add_argument('--xml', default=None, help='the xml configuration file')
parser.add_argument('--demo_files', nargs='+', help='the environment used for test')
parser.add_argument('--ratios', nargs='+', type=float, help='the ratio of demos to load')
parser.add_argument('--eval_epochs', type=int, default=10, help='the epochs for evaluation')
parser.add_argument('--feasibility_model', default=None, nargs='+', help='the path to the feasibility model')
parser.add_argument('--mode', help='the mode of feasibility')
parser.add_argument('--n_domains', type=int, default=4, help='the number of domains')
parser.add_argument('--begin-index', type=int, default=0, help='Number of cluster to begin')
parser.add_argument('--cluster_list', nargs='+', help='the cluster used for test')
parser.add_argument('--dataset', type=str, default=None, help='the root of data path')
args = parser.parse_args()
# from tensorboardX import SummaryWriter
from logger import *
import json
import copy
# re-define datasets
cluster_list = args.cluster_list
logger = CompleteLogger('log/'+ args.env_name + '/'+ args.mode + '/target-' + os.path.splitext(args.xml)[0] + '_N' + str(args.n_domains) +\
 '_ratio_{}'.format(str([args.ratios[0]])))

# writer=SummaryWriter(log_dir=logger.root + '/runs')
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
wandb.run.name = f"{args.env_name}-target-{os.path.splitext(args.xml)[0]}"+ \
        now
wandb.config.update(args)

json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)
save_path = logger.get_checkpoint_path('seed_{}_gail_model_disc3'.format(args.seed))
env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False)
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.feasibility_model is not None:
    expert_pairs, expert_trajs, pair_traj_id, traj_traj_id,  count_of_traj = load_demos(args.demo_files, ratios=args.ratios, cluster_list=args.cluster_list)
    agents = []
    value_nets = []
    discriminators = []
    model_dict_list = []

    for clu_i in range(len(cluster_list)):
        begin_idx = cluster_list[clu_i]
        save_policy_file = args.feasibility_model[clu_i]
        try:
            model_dict = torch.load(save_policy_file)
            print(save_policy_file, model_dict.keys())
            model_dict_list.append(model_dict)
        except:
            print('disc does not exist for cluster: ', save_policy_file)
    for model_dict in model_dict_list:
        
        single_policy = Policy(num_inputs, num_actions, args.hidden_dim)
        single_value_net = Value(num_inputs, args.hidden_dim)
        single_discriminator = Discriminator(num_inputs + num_inputs, args.hidden_dim).to(device)

        single_policy.load_state_dict(model_dict['policy'])
        single_value_net.load_state_dict(model_dict['value'])
        single_discriminator.load_state_dict(model_dict['disc'])

        agents.append(single_policy)
        value_nets.append(single_value_net)
        discriminators.append(single_discriminator)

    expert_traj_pairs_all = []
    all_len_ = 0
    expert_traj = np.concatenate(expert_pairs, axis=0)
    for i in range(len(expert_trajs)):
        singel_traj_pairs = expert_traj[all_len_:all_len_ + len(expert_trajs[i]) - 1]
        all_len_ += (len(expert_trajs[i])- 1)
        expert_traj_pairs_all.append(singel_traj_pairs)

    feasibility_traj = compute_feasibility_traj_discri(expert_traj_pairs_all, expert_trajs, traj_traj_id, discriminators, device=device)
    feasibility = feasibility_traj[pair_traj_id]
    ########### evaluate feasibility ################
    all_len = 0
    for t in range(len(agents)):
        print("mean feas {}".format(t), feasibility_traj[all_len:all_len + int(count_of_traj[t])].mean())
        all_len += int(count_of_traj[t])
    all_len = 0
    for t in range(len(agents)):
        print("sum feas {}".format(t), feasibility_traj[all_len:all_len + int(count_of_traj[t])].sum())
        all_len += int(count_of_traj[t])
else:
    print('=> run baseline')
    expert_pairs = load_demos(args.demo_files, ratios=args.ratios)[0]
    feasibility = np.ones(sum([expert_traj.shape[0] for expert_traj in expert_pairs]))
expert_traj = np.concatenate(expert_pairs, axis=0)

policy_net = Policy(num_inputs, num_actions, args.hidden_dim)#.to(device)
value_net = Value(num_inputs, args.hidden_dim).to(device)
discriminator = Discriminator(num_inputs + num_inputs, args.hidden_dim).to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)


def expert_reward(states, actions):
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    with torch.no_grad():
        state_action = torch.Tensor(np.concatenate([states, actions], 1)).to(device)
        return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()


def evaluate(episode, best_reward):
    env.seed(1234)
    with torch.no_grad():
        avg_reward = 0.0
        frames = []
        for epo in range(args.eval_epochs):
            state = env.reset()
            for _ in range(10000): # Don't infinite loop while learning
                state = torch.from_numpy(state).unsqueeze(0)
                action, _, _ = policy_net(Variable(state))
                action = action.data[0].numpy()
                if args.render and epo == 0 and episode % 500 == 0:
                    frames.append(np.transpose(env.render(mode='rgb_array', height=256, width=256), (2, 0, 1)))
                next_state, reward, done, _ = env.step(action)
                avg_reward += reward
                if done:
                    break
                state = next_state
        all_avg_reward = avg_reward / args.eval_epochs
        wandb.log({
            "output/Average Reward": all_avg_reward
        })
        instant_save_path = save_path.replace('.pth', f'_step{episode}.pth')
        torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(),'disc':discriminator.state_dict(), 'rew':all_avg_reward}, instant_save_path)
        print("=> save model with reward: ", all_avg_reward, " at ", instant_save_path)
        if args.render and episode % 500 == 0:
            wandb.log({f"video/episode{episode}": wandb.Video(np.array(frames), fps=120)})
        print('Evaluation: Episode ', episode, ' Reward ', all_avg_reward)
        if best_reward < all_avg_reward:
            best_reward = all_avg_reward
            torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(),'disc':discriminator.state_dict(), 'rew':best_reward}, save_path)
            print("=> save model with best reward: ", best_reward, " at ", save_path)
    return best_reward

all_idx = np.arange(0, expert_traj.shape[0])
p_idx = np.random.permutation(expert_traj.shape[0])
expert_traj = expert_traj[p_idx, :]
feasibility = feasibility[p_idx]

feasibility = feasibility / (np.sum(feasibility)+0.0000001)
feasibility[0] = 1 - np.sum(feasibility[1:])

best_reward = -1000000

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

    while num_steps < args.batch_size:
        state = env.reset()
   

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

    if i_episode % args.eval_interval == 0:
        best_reward = evaluate(i_episode, best_reward)

    rewards = expert_reward(states, next_states)
    for idx in range(len(states)):
        memory.push(states[idx][0], actions[idx], mem_mask[idx], mem_next[idx], \
                    rewards[idx][0])
    batch = memory.sample()
    update_params(batch, policy_net=policy_net, value_net=value_net, value_optimizer=value_optimizer,
                    value_criterion=value_criterion, args=args)

    ### update discriminator ###
    next_states = torch.from_numpy(np.concatenate(next_states))
    states = torch.from_numpy(np.concatenate(states))
   

    labeled_num = min(expert_traj.shape[0], num_steps)

    idx = np.random.choice(all_idx, labeled_num, p=feasibility.reshape(-1))

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
    ############# logging ###############
    if i_episode % args.log_interval == 0:
        print('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))
        wandb.log({
            'output/Mean train reward: ': np.mean(reward_batch),
            'output/Max train reward: ': max(reward_batch)
        })

    wandb.log({
        'debug/discriminator real: ': torch.mean(real),
        'debug/disctiminator fake: ': torch.mean(fake),
        'debug/disctiminator loss: ': disc_loss,
        'debug/r(s, s_hat)': np.mean(rewards)
    })
