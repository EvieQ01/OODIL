import pdb
from tqdm import tqdm
import numpy as np
import torch
import wandb
from  matplotlib import pyplot as plt
import time
from pdenv.gym_panda.envs.panda_env import PandaEnv

from .buffer import Buffer
import torch.nn as nn
from gail_airl_ppo.network import StateIndependentPolicy
def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def get_delta_desired(env):
    ''''
    set desired posi to obs_posi
    '''
    panda_posi = env.panda.state['ee_position'] #(3)
    obj_posi = env.get_obj_ee_posi()      #(3)
    desire_posi = env.panda.desired['ee_position'] 
    action =  (obj_posi - desire_posi- np.array([.1, 0.,-0.0])) * 240 #0 #40
    # adjust y, z first!
    if obj_posi[1] - desire_posi[1] > 0.05 or (obj_posi[2] - desire_posi[2] > 0.05):
        action *= np.array([0.,1.,1.])
    action[0] = 1* action[0] if action[0] > 0 else action[0]
    action = np.clip(action, -1.0, 1.0)
    return action 

def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0,render=False,args=None):
    if render:
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        wandb.init(project="robot_test", entity="your_name")
        wandb.run.name = f"{args.env_id}"+ now
        wandb.config.update(args)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    frames = []
    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0
    episode_step = 0
    
    if algo is not None and algo != 'random':
        actor = StateIndependentPolicy(
            state_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            hidden_units=(128, 128),
            hidden_activation=nn.Tanh()
        ).to(device)
        state_dict = torch.load(algo)
        actor.load_state_dict(state_dict=state_dict)
        
    for _ in tqdm(range(2, buffer_size * 2)):
        t += 1
        
        # random explore
        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            # gail explore
            if algo is not None:

                state_torch = torch.tensor(state, dtype=torch.float, device=device)
                with torch.no_grad():
                    action, log_pi = actor.sample(state_torch.unsqueeze_(0))

            else:
                action = get_delta_desired(env)
        
        next_state, reward, done, _ = env.step(action)
        if render and episode_step % 20 == 0:
            frames.append(np.transpose(env.render(mode='rgb_array'), (2, 0, 1)))
        episode_step += 1
        mask = False if t == env._max_episode_steps else done

        if algo is not None: # for id baseline 23
            buffer.append(state, np.array(action.squeeze()), reward, mask, next_state)
        else:
            buffer.append(state, action, reward, mask, next_state)
        episode_return += reward
        if episode_step == env._max_episode_steps and algo is None: # discard!
            buffer.remove(episode_step)
            if render:
                wandb.log({f"video/test_video": wandb.Video(np.array(frames), fps=1200)})
                render=False # only render the first video
            print(f'not done with state: {state}\t steps [{episode_step}]')
            done = True
        if done:
            state = env.reset()
            t = 0
            if render:
                wandb.log({f"video/test_video": wandb.Video(np.array(frames), fps=1200)})
                render=False # only render the first video
            if episode_return > 1000 or algo is not None: #only when algo is None we limit it to expert
                buffer.add_traj_length(episode_step)
                num_episodes += 1
                total_return += episode_return
                print('episode return: ', episode_return)
            episode_return = 0.0
            episode_step = 0
            if render:
                wandb.log({f"video/test_video": wandb.Video(np.array(frames), fps=1200)})
                frames = []
                # render=False # only render the first video
            continue
        if buffer._p == buffer_size - 1:
            break

        state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    print('sum buffer traj length:', sum(buffer.traj_len))
    if args.render:
        plt.figure()
        plt.scatter(x=buffer.states[:sum(buffer.traj_len), 0], y=buffer.states[:sum(buffer.traj_len), 1], s=1)
        plt.xlim(-1, 1)  # 设置x轴的数值显示范围
        plt.ylim(-1, 1)  # 设置y轴的数值显示范围
        plt.savefig(f'temp_episode.png')
        wandb.log({f"image/all_traj": wandb.Image(f'temp_episode.png')})


    return buffer
