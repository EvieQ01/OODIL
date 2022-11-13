from tqdm import trange
import argparse

import gym
import gym.spaces
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import Value, Policy, Discriminator
from replay_memory import Memory
from utils import *
from loss import *
from feasibility_utils import compute_feasibility_traj_discri
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

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = False
# use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from RL_utils import update_params, select_action
from carlo_utils import make_observation_norm, evaluate
parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', type=str, default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
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
parser.add_argument('--demo_files', nargs='+', help='the environment used for test')
parser.add_argument('--ratios', nargs='+', type=float, help='the ratio of demos to load')
parser.add_argument('--eval_epochs', type=int, default=10, help='the epochs for evaluation')
parser.add_argument('--save_path', help='the path to save model')
parser.add_argument('--feasibility_model', default=None, nargs='+', help='the path to the feasibility model')
parser.add_argument('--mode', help='the mode of feasibility')
parser.add_argument('--cluster_list', nargs='+', help='the cluster used for test')
parser.add_argument('--init_range', nargs='+', help='the range of init obs.x', default=None)
parser.add_argument('--dataset', type=str, default='../demo', help='the  root path of dataset')
parser.add_argument('--load_path', type=str, default=None, help='the path of saved model')
args = parser.parse_args()
args.init_range = [int(i) for i in args.init_range]
from logger import *
import json

logger = CompleteLogger('log/'+ args.env_name + '/3GAIL_'+ args.mode +\
 '_ratio_{}'.format(str([args.ratios])))

now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
wandb.init(project="CARLO_3GAIL_disc")
wandb.run.name = f"{args.env_name}"+f"_N{str(len(args.demo_files))}" +\
        now
wandb.config.update(args)

json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)

save_path = logger.get_checkpoint_path('seed_{}_gail_model_disc'.format(args.seed))

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print(f"=> num_actions: {num_actions}", f"=> num_states: {num_inputs}")

def load_demos(demo_files, ratios):
    state_files = []
    trajs = []
    traj_traj_id = []
    traj_id = 0
    pair_traj_id = []
    init_obs = []
    all_mapping = None
    count_of_traj = []
    actions = []
    clu_id = 0
    for i in range(len(args.cluster_list)):
        state_pairs = []
        demo_file = demo_files[0].replace('simclr_dcn_0',f'simclr_dcn_{args.cluster_list[i]}')
        if 'storage'  in args.dataset:
            demo_file = os.path.join(args.dataset, demo_file)

        try:
            raw_demos = pickle.load(open(demo_file, 'rb'))
        except:
            print("empty cluster[loading fail]: ", demo_file)
            continue
        use_num = int(len(raw_demos['obs'])*ratios[0])
        current_state = raw_demos['obs'][0:use_num]
        print("=> use demo {}:".format(demo_file), len(current_state))
        count_of_traj.append(len(current_state))
        trajs += [np.array(traj)[:, :num_inputs] for traj in current_state]# leave out the last obs dim
        # 0 0 0 ..., 1 1 1, ..., 2,2,2, ..., 200, 200, 200
        traj_traj_id += [clu_id]*len(current_state)
        clu_id += 1
        for j in range(len(current_state)):
            current_state[j] = np.array(current_state[j])[:, :num_inputs]# leave out the last obs dim
            state_pairs.append(np.concatenate([np.array(current_state[j][:-1, :]), np.array(current_state[j][1:, :])], axis=1))
            pair_traj_id.append(np.array([traj_id]*np.array(current_state[j]).shape[0]))
            traj_id += 1
        state_files.append(np.concatenate(state_pairs, axis=0))
    return state_files, trajs, np.concatenate(pair_traj_id, axis=0), np.array(traj_traj_id), init_obs, all_mapping, count_of_traj, actions

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.feasibility_model is not None:
    expert_pairs, expert_trajs, pair_traj_id, traj_traj_id, init_obs, all_mapping, count_of_traj, actions = load_demos(args.demo_files, args.ratios)
    agents = []
    value_nets = []
    discriminators = []
    model_dict_list = []
    save_policy_file = args.feasibility_model
    for clu in args.cluster_list:
        current_save_policy_file = args.feasibility_model[0].replace('index_0', f'index_{clu}')
        try:
            model_dict = torch.load(current_save_policy_file)
            print(current_save_policy_file, model_dict.keys())
            model_dict_list.append(model_dict)
        except:
            print('disc does not exist for cluster: ', current_save_policy_file)
    for model_dict in model_dict_list:
        
        single_policy = Policy(num_inputs, num_actions, args.hidden_dim)
        single_value_net = Value(num_inputs, args.hidden_dim)#.to(device)
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
    #### norm ############## (s, s') # (50000 * 14)
    expert_traj =  make_observation_norm(expert_traj, env, num_inputs)
    for i in range(len(expert_trajs)):
        singel_traj_pairs = expert_traj[all_len_:all_len_ + len(expert_trajs[i]) - 1]
        all_len_ += (len(expert_trajs[i])- 1)
        expert_traj_pairs_all.append(singel_traj_pairs)
    
    feasibility_traj = compute_feasibility_traj_discri(expert_traj_pairs_all, expert_trajs, traj_traj_id, agents, discriminators, \
             env, count_of_traj=count_of_traj, init_obs=init_obs, args=args, device=device)
    feasibility = feasibility_traj[pair_traj_id]
    step_count = [len(expert_trajs[i])-1 for i in range(len(expert_trajs))]
    ########### evaluate feasibility ################
    all_len = 0
    for t in range(len(agents)):
        print("mean feas {}".format(t), feasibility_traj[all_len:all_len + int(count_of_traj[t])].mean())
        all_len += int(count_of_traj[t])

elif args.load_path is None:
    print('=> run baseline')
    expert_pairs = load_demos(args.demo_files, args.ratios)[0]
    feasibility = np.ones(sum([expert_traj.shape[0] for expert_traj in expert_pairs]))

    expert_traj = np.concatenate(expert_pairs, axis=0)
    #### norm ############## (s, s') # (50000 * 14)
    expert_traj =  make_observation_norm(expert_traj, env, num_inputs)

policy_net = Policy(num_inputs, num_actions, args.hidden_dim)#.to(device)
value_net = Value(num_inputs, args.hidden_dim).to(device)
discriminator = Discriminator(num_inputs + num_inputs, args.hidden_dim).to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)
if args.load_path is not None: # load pretrained model
    print("=> load: ",args.load_path)
    model_dict = torch.load(args.load_path)
    policy_net.load_state_dict(model_dict['policy'])
    best_reward = evaluate(env=env, policy_net=policy_net, value_net=value_net, discriminator=discriminator, episode=0, best_reward=0,\
                        save_path=save_path, args=args, all_trajs=None)
    exit(0)

def expert_reward(states, actions):
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    with torch.no_grad():
        state_action = torch.Tensor(np.concatenate([states, actions], 1)).to(device)
        return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()

# 
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

    time_sample = time.time()
    while num_steps < args.batch_size:
        state = env.reset(init_range=args.init_range)
   

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
        best_reward = evaluate(env=env, policy_net=policy_net, value_net=value_net, discriminator=discriminator, episode=i_episode, best_reward=best_reward,\
                            save_path=save_path, args=args, all_trajs=None)

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
    time_train_disc = time.time() - time_train_disc
    ############################

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
        'debug/r(s, s_hat)': np.mean(rewards),
        'Timing/rollout': time_sample,
        'Timing/RL': time_update_param,
        'Timing/IRL:train discri': time_train_disc,
    })
