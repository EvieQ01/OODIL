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
from models.old_models import *
from replay_memory import Memory
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from loss import *

import numpy as np

import time
import sys
sys.path.append('../all_envs')

import hopper
import walker
import halfcheetah

import pickle
import wandb
from RL_utils import update_params, select_action
from demo_utils import load_demos
wandb.init(project="DifferentDynamics_GAIL2")

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"

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
parser.add_argument('--xml', default=None, help='the xml configuration file')
parser.add_argument('--demo_files', nargs='+', help='the environment used for test')
parser.add_argument('--test_demo_files', nargs='+', help='the environment used for test')
parser.add_argument('--ratios', nargs='+', type=float, help='the ratio of demos to load')
parser.add_argument('--eval_epochs', type=int, default=10, help='the epochs for evaluation')
parser.add_argument('--feasibility_model', default=None, help='the path to the feasibility model')
parser.add_argument('--mode', help='the mode of feasibility')
parser.add_argument('--begin-index', type=int, default=10, help='the index of cluster')
parser.add_argument('--dataset', type=str, default='../demo', help='the source of data root')
args = parser.parse_args()
from tensorboardX import SummaryWriter
from logger import *
import json
import copy
# re-define datasets
logger = CompleteLogger('log/'+ args.env_name + '/'+ args.mode + '/target-' + os.path.splitext(args.xml)[0] +\
    '_ratio_{}'.format(str(args.ratios)))
# writer=SummaryWriter(log_dir=logger.root + '/runs')
now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
wandb.config.update(args)
wandb.run.name = f"{args.env_name}_{os.path.splitext(args.xml)[0]}_dcn" +\
        '_cluster' + str(args.begin_index) + now

json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)
save_policy_file = logger.get_checkpoint_path('seed_{}_gail_model_begin_index_{}'.format(args.seed, args.begin_index))

demos = []
env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False) 
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]


env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

expert_pairs = load_demos(args.demo_files, args.ratios, begin_index=args.begin_index)[0]
if len(expert_pairs) == 0:
    exit(0) # empty clu
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
            for t in range(10000): # Don't infinite loop while learning
                state = torch.from_numpy(state).unsqueeze(0)
                action, _, _ = policy_net(Variable(state))
                action = action.data[0].numpy()
                next_state, (reward), done, _ = env.step(action)
                if args.render and epo == 0 and episode % 500 == 0:
                    frames.append(np.transpose(env.render(mode='rgb_array', height=256, width=256), (2, 0, 1)))
                avg_reward += reward
                if done:
                    break
                state = next_state
            # axes are (time, channel, height, width)
            # frames = np.random.randint(low=0, high=256, size=(10, 3, 100, 100), dtype=np.uint8)
            if args.render and epo == 0 and episode % 500 == 0:
                wandb.log({f"video/episode{episode}": wandb.Video(np.array(frames), fps=120)})
        print('Evaluation: Episode ', episode, ' Reward ', avg_reward / args.eval_epochs)
        wandb.log(
            {'output/Average Reward': avg_reward / args.eval_epochs}
        )
        current_path = save_policy_file.replace('.pth', f'_step{episode}.pth')
        torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(),'disc':discriminator.state_dict(), 'rew':best_reward}, current_path)
        print('=> save current policy at: ', current_path)
        if best_reward < avg_reward / args.eval_epochs:
            best_reward = avg_reward / args.eval_epochs
            torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(),'disc':discriminator.state_dict(), 'rew':best_reward}, save_policy_file)
            print('=> save best policy at: ', save_policy_file)
    return best_reward

all_idx = np.arange(0, expert_traj.shape[0])
p_idx = np.random.permutation(expert_traj.shape[0])
expert_traj = expert_traj[p_idx, :]

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
            next_state, reward_step, done, _ = env.step(action)
            next_states.append(np.array([next_state]))
            reward_sum += reward_step #* (args.discount ** t)

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
    wandb.log({
        'debug/discriminator real: ': torch.mean(real),
        'debug/disctiminator fake: ': torch.mean(fake),
        'debug/disctiminator loss: ': disc_loss,
        'debug/r(s, s_1): ': np.mean(rewards)
    })
    ############################
    if i_episode % args.log_interval == 0:
        print('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))
        wandb.log({
            'output/Total train reward: ': np.mean(reward_batch),
            'output/Max train reward: ': max(reward_batch)
        })
