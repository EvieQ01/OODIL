import time
import os
import argparse
from datetime import datetime
import torch
import pdenv.gym_panda

from gail_airl_ppo.env import make_env
from gail_airl_ppo.buffer import SerializedBuffer
from gail_airl_ppo.algo import ALGOS
from gail_airl_ppo.trainer import Trainer
import gym
import wandb
from gym import error, spaces, utils
wandb.init(project="Robot_feas2_gail", entity="qiuyiwen")
import numpy as np
def run(args):
    env = gym.make(args.env_id)
    if args.light_obj:
        env.set_obj(mass=0.001)

    env_test = env
    env.set_action_mode(mode=0)
    env.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(env.n, ), dtype=np.float32)
    env.panda.init_range = args.init_range

    ## define buffer path
    buffer_path = os.path.join(args.dataset,args.root_path,args.buffer)
    print("=> use data buffer: ", buffer_path)
    buffer_exp = SerializedBuffer(
        path=buffer_path,
        device=torch.device("cuda" if args.cuda else "cpu")
    )
    print("=> use data: ", len(buffer_exp.traj_len))

    algo = ALGOS[args.algo](
        buffer_exp=buffer_exp,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=torch.device("cuda" if args.cuda else "cpu"),
        seed=args.seed,
        rollout_length=args.rollout_length,
        use_minibatch=args.use_minibatch,
        epoch_ppo=args.epoch_ppo,
        epoch_disc=args.epoch_disc
   )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'{args.mode}-seed{args.seed}-{time}-cluster-{args.begin_idx}')

    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seed=args.seed,
        render=args.render
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root_path', type=str, default='panda-v0')
    p.add_argument('--buffer', type=str, default='size100000_able_panda_init0.5.pth')
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=3 * 10**7)
    p.add_argument('--eval_interval', type=int, default=5 * 10**5)
    p.add_argument('--env_id', type=str, default='panda-v0')
    p.add_argument('--algo', type=str, default='gailfo')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init_range', type=float, default=2.0)

    # logging
    p.add_argument('--render',action='store_true')

    # dataset
    p.add_argument('--begin_idx', type=int, default=0)
    p.add_argument('--mode', type=str, default='resplit_simclr') # baseline, resplit_simclr
    p.add_argument('--dataset', default='buffers') # baseline, resplit_simclr
    p.add_argument('--use_minibatch', default=False, action='store_true')    
    p.add_argument('--light_obj', default=False, action='store_true')    
    p.add_argument('--epoch_ppo', type=int, default=10)
    p.add_argument('--epoch_disc', type=int, default=20)
    
    args = p.parse_args()
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    
    if 'split' in args.mode:
        args.buffer = args.buffer.replace('split0', f'split{args.begin_idx}')
        print("Warning, change buffer according to args.begin_idx => ", args.buffer)

    wandb.config.update(args)
    wandb.run.name = f"target_{args.env_id}_source_{args.root_path}_cluster{args.begin_idx}_{args.buffer}_" + now

    run(args)
