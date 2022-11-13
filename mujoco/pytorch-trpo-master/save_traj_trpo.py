# import free_mjc
import pdb
import sys
from tkinter.messagebox import NO
from logger import *
import json
sys.path.append('../all_envs')
import swimmer
import walker
import halfcheetah
import humanoid
import ant
import hopper
from running_state import ZFilter

# import envs.swimmer
# import envs.ant
# import envs.params.swimmer
# import envs.params.hopper
# import envs.params.half_cheetah
# import envs.params.walker2d
# from utils import *
from itertools import count
import argparse
import gym
import os
import sys
import pickle
import imageio
from tqdm import trange
import matplotlib.pyplot as plt
import torch
import numpy as np
from models import Policy, Value
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--explore', action='store_true', default=False,
                    help='whether use expert policy')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-episode', type=int, default=10000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
parser.add_argument('--dump', default=False, action='store_true')
parser.add_argument('--xml', type=str, default='', metavar='N',
                    help='xml of env')
parser.add_argument('--gail_model_path', type=str, default=None, metavar='N',
                    help='model path of Gail pretrained model')
parser.add_argument('--count_of_demos', type=int, default=500,
                    help='how many demonstrations to save')
parser.add_argument('--id', type=int, default=0,
                    help='how many demonstrations to save')
parser.add_argument('--total_steps', type=int, default=50000,
                    help='how many total steps to save, useful when policy is not optimal and hard to limit step in each episode')
                    # default as 1/10 of 1000 * 500 steps
args = parser.parse_args()
logger = CompleteLogger('log/'+ args.env_name + '/'+ os.path.splitext(args.xml)[0] + '_save_traj')
json.dump(vars(args), logger.get_args_file(), sort_keys=True, indent=4)

# creat envs
dtype = torch.float32
torch.set_default_dtype(dtype)
env = gym.make(args.env_name, xml_file=args.xml, exclude_current_positions_from_observation=False)
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
print('state dim:', state_dim)
print('action dim:', num_actions)

# load models
save_demo_dir = '../demo/' + os.path.splitext(args.xml)[0]
if 'Swimmer' in args.env_name and os.path.splitext(args.xml)[0] != 'swimmer':
    save_demo_path = '../demo/swimmer_' + os.path.splitext(args.xml)[0].replace('leg_', '')  + '/batch_00.pkl'

saved_path = '../checkpoints/' + os.path.splitext(args.xml)[0]  + '_model.pth'
if args.explore:
    save_demo_path = '../demo/' + os.path.splitext(args.xml)[0]  + f'/explore_batch_{args.id}.pkl'
if args.gail_model_path:
    save_demo_path = '../demo/' + os.path.splitext(args.xml)[0]  + f'/gail_explore_batch_{args.id}.pkl'
    save_demo_path = '../demo/' + os.path.splitext(args.xml)[0]  + '/batch_01.pkl'
if not os.path.exists(save_demo_dir):
    os.mkdir(save_demo_dir)
############ TODO

# ######### load ###########
policy_net = Policy(num_inputs, num_actions)
if not args.explore:
    print('use expert policy')
    if args.gail_model_path is not None:
        saved_path = args.gail_model_path
        state_dict =  torch.load(saved_path, map_location='cpu')
        policy_net.load_state_dict(state_dict['policy'])
        running_state = ZFilter((num_inputs,), clip=5)
    else:
        state_dict =  torch.load(saved_path, map_location='cpu')
        policy_net.load_state_dict(state_dict)
        print('use expert running state') # using running state
        running_state = torch.load("../checkpoints/running_state_{}".format(args.xml))
else:
    running_state = ZFilter((num_inputs,), clip=5)

raw_demos = {}
def main_loop():

    num_steps = 0
    
    raw_demos['obs'] = []
    raw_demos['obs_normed'] = []
    raw_demos['next_obs'] = []
    raw_demos['action'] = []
    for i_episode in count():
        expert_traj = []
        expert_traj_normed = []
        actions = []
        state = env.reset()
        # state = torch.tensor(running_state(state, update=False)).unsqueeze(0).to(dtype)
        reward_episode = 0
        episode_steps = 1

        rewards = []
        frames = []
        reward_sum = 0.
        expert_traj.append(state)
        for t in trange(50000):

            action = policy_net(torch.from_numpy(state.astype(np.float32)).unsqueeze(0))
            action = (torch.normal(action[0], action[2])).detach().numpy() # norm action
            if len(action.shape) > 1:
                action = action.squeeze(0)
            next_state, reward, done, _ = env.step(action)
            expert_traj.append(next_state)
            actions.append(action)
            reward_sum += reward

            next_state = running_state(next_state, update=False)
            reward_episode += reward
            rewards.append(reward)
            episode_steps += 1

            if args.render and i_episode % 200 == 0 :
                frames.append(env.render(mode='rgb_array', height=2048, width=2048))
            if done:
                if episode_steps >= 1000 or args.explore or args.gail_model_path is not None: # 1000 steps -> done
                    print("steps in this episode: ",episode_steps)
                    print(f"total steps: {num_steps}/{args.total_steps}")
                    print("save traj[{}] with rewards: {}".format(len(raw_demos['obs']), reward_episode))
                    raw_demos['obs'].append(expert_traj)
                    raw_demos['next_obs'].append(expert_traj[1:] + next_state)
                    raw_demos['action'].append(actions)
                    num_steps += episode_steps
                    if len(raw_demos['obs']) == 1:
                        print("No.1 traj:", raw_demos['obs'][0][0])
                    if len(raw_demos['obs']) == args.count_of_demos and num_steps >= args.total_steps:
                        return
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
        if args.render and i_episode % 200 == 0:
            imageio.mimsave(logger.get_image_path(f'demo_{i_episode}_{args.xml}.mp4'), frames, fps=120)
            plt.clf()
            plt.plot(rewards)
            plt.savefig(f'demo_{i_episode}.png')

        if i_episode >= args.max_expert_episode:
            break


main_loop()
if args.dump:
    pickle.dump(raw_demos,
                open(save_demo_path, 'wb'))
