import os
import argparse
import torch
import pdenv.gym_panda
from gail_airl_ppo.env import make_env
from gail_airl_ppo.algo import SACExpert
from gail_airl_ppo.panda_utils import collect_demo
import gym
from gym import error, spaces
import numpy as np

def run(args):
    
    env = gym.make(args.env_id)
    # use xyz to make demonstration.
    if args.explore_model == 'random':
        args.p_rand = 1.
        mode = 0 # use for id baseline2
        file_name = f'size{args.buffer_size}_panda_init{args.init_range}_index{args.index}_random_explore_new.pth' 

    elif args.explore_model is None: # expert
        file_name = f'size{args.buffer_size}_panda_init{args.init_range}_new.pth' 
        mode = 1
    else:  # use for id baseline3, gail explore
        mode = 0
        file_name = f'size{args.buffer_size}_panda_init{args.init_range}_index{args.index}_gail_explore_new.pth' 
    env.set_action_mode(mode=mode)
    env.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(env.n, ), dtype=np.float32)
    env.panda.init_range = args.init_range
    buffer = collect_demo(
        env=env,
        algo=args.explore_model,
        buffer_size=args.buffer_size,
        device=torch.device("cuda" if args.cuda else "cpu"),
        std=args.std,
        p_rand=args.p_rand,
        seed=args.seed,
        render=args.render,
        args=args
    )
    buffer.save(os.path.join(
                            'buffers10.0',
                            args.env_id,
                            file_name
                        ), true_length=sum(buffer.traj_len))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--weight', type=str, required=True)
    p.add_argument('--env_id', type=str, default='panda-v2')
    p.add_argument('--buffer_size', type=int, default=10**4)
    p.add_argument('--std', type=float, default=0.0)
    p.add_argument('--p_rand', type=float, default=0.0)
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init_range', type=float, default=2.0)
    p.add_argument('--render', action='store_true')

    p.add_argument('--explore_model', default=None)
    p.add_argument('--index', default=0, type=int)
    # None means expert policy
    # 'random' meand random explore
    # 'model_path_to_gail_model' to do gail explore.


    # # extra params
    # p.add_argument('--init_range', type=float, default=0.5)
    args = p.parse_args()
    run(args)
