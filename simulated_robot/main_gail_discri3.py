import pdb
import time
import os
import argparse
from datetime import datetime
import torch
import pdenv.gym_panda

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import ConcatStateBuffers, SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer

from gail_airl_ppo.network import GAILDiscrim
import gym
import wandb
from gym import error, spaces, utils
import torch.nn as nn

from feasibility_utils import compute_feasibility_traj_discri
import numpy as np
from gym.vector.async_vector_env import AsyncVectorEnv
def run(args):

    def gen_env(seed=0):
        env = gym.make(args.env_id)
        if args.light_obj:
            env.set_obj(mass=0.001)
        env.set_action_mode(mode=0)
        env.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(env.n, ), dtype=np.float32)
        env.panda.init_range = args.init_range
        env.seed(seed)
        return env

    if args.use_vectorenv:
        env = AsyncVectorEnv([lambda: gen_env(args.seed+_) for _ in range(args.num_workers)])
        env_test = gen_env((2**31)-args.seed)
    else:
        env = gen_env(args.seed)
        env_test = env

    ## define buffer path
    if len(args.buffers) == 1:
        buffer_path = os.path.join(args.dataset,args.buffers[0])
        buffer_exp = SerializedBuffer(
            path=buffer_path,
            device=torch.device("cuda" if args.cuda else "cpu")
        )
    else:
        ## concat dataset
        all_buffers = []
        for demo_file in args.buffers:
            buffer_exp = SerializedBuffer(
                path=os.path.join(args.dataset, demo_file),
                device=torch.device("cuda" if args.cuda else "cpu"))
            all_buffers.append(buffer_exp)
            print(f"=>load from {demo_file}, {len(buffer_exp.traj_len)} trajs, {sum(buffer_exp.traj_len)} state pairs in all")
        buffer_exp = ConcatStateBuffers(all_buffers)
    if args.disc_model_paths is not None:
        ## load models:
        discriminators = []
        for disc_model_path in args.disc_model_paths:
            state_dict = torch.load(disc_model_path)
            disc = GAILDiscrim(
                state_shape=env.observation_space.shape,
                action_shape=env.observation_space.shape, # action is next_state!!
                hidden_units=(args.hidden_units, args.hidden_units),
                hidden_activation=nn.Tanh()
            ).to(torch.device("cuda" if args.cuda else "cpu"))

            disc.load_state_dict(state_dict=state_dict)
            discriminators.append(disc)
        # feasibility for each trajectory
        feasibility = compute_feasibility_traj_discri(buffer_exp, discris=discriminators, device=torch.device("cuda" if args.cuda else "cpu"))
        buffer_exp.reweight_for_each_traj(feasibility)
    else:
        feasibility = np.ones(len(buffer_exp.traj_len))

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env_test.observation_space.shape,
        action_shape=env_test.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        use_minibatch=args.use_minibatch,
        epoch_ppo=args.epoch_ppo,
        epoch_disc=args.epoch_disc,
        use_vectorenv=args.use_vectorenv
            )


    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'{args.mode}-seed{args.seed}-{time}-gail-weighted-{args.source_str}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        render=args.render,
        render_fix=args.render_fix
    )
    if args.load_path is None:
        if args.use_vectorenv:
            trainer.train_vector(args.num_workers, max_episode_steps=env_test._max_episode_steps)
        else:
            trainer.train()
    else:
        state_dict = torch.load(args.load_path)
        trainer.algo.actor.load_state_dict(state_dict)
        trainer.evaluate(0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # p.add_argument('--root_paths', nargs='+', default='panda-v0')
    p.add_argument('--buffers', nargs='+', default='size100000_able_panda_init0.5.pth')
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=5 * 10**7)
    p.add_argument('--eval_interval', type=int, default=5 * 10**5)
    p.add_argument('--env_id', type=str, default='panda-v0')
    p.add_argument('--algo', type=str, default='gailfo')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init_range', type=float, default=2.0)

    # logging
    p.add_argument('--render',action='store_true')
    p.add_argument('--render_fix',action='store_true')
    p.add_argument('--source_str', type=str, default='from_4_6_0') # baseline, resplit_simclr

    # dataset
    p.add_argument('--begin_idx', type=int, default=0)
    p.add_argument('--mode', type=str, default='resplit_simclr') # baseline, resplit_simclr
    
    # saved path of cluster discriminators
    p.add_argument('--cluster_list', nargs='+')
    p.add_argument('--disc_model_paths', nargs='+',default=None) # a single model is ok, then change according to cluster_list
    p.add_argument('--dataset', default='buffers') # baseline, resplit_simclr
    p.add_argument('--load_path', default=None) # baseline, resplit_simclr

    p.add_argument('--use_minibatch', default=False, action='store_true')    
    p.add_argument('--epoch_ppo', type=int, default=10)
    p.add_argument('--epoch_disc', type=int, default=20)
    p.add_argument('--use_vectorenv', default=False, action='store_true')
    p.add_argument('--light_obj', default=False, action='store_true')    
    p.add_argument('--num_workers', type=int, default=4, choices=[2,4,8,16])
    p.add_argument('--hidden_units', type=int, default=256)
    args = p.parse_args()

    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    if 'split' in args.mode:
        args.buffers = [args.buffers[0].replace('split0', f'split{int(i)}') for i in args.cluster_list]
        print("Warning, change buffer according to cluster_list => ", args.buffers)
    print("Warning, change disc_model_paths should be according to cluster_list => ", args.disc_model_paths)

    wandb.init(project="Robot_disc3_gail", entity="qiuyiwen")
    wandb.run.name = f"target_{args.env_id}_source_{args.buffers[0]}_cluster{args.cluster_list}" + now
    wandb.config.update(args)

    run(args)
