from random import randint
from torch.autograd import Variable
import numpy as np
import wandb
import matplotlib.pyplot as plt
import torch
import time
import pdb
def select_init_obs(all_trajs):
    """
    all_trajs shape as n_samples * 1000steps * obs_dim
    """
    sample_idx = randint(0, len(all_trajs) - 1)
    return all_trajs[sample_idx][0, :]

def make_observation_norm(expert_traj, env, num_inputs):
    '''
    expert_traj shape as (50000 * 14)
    '''
    print("=> make observation_norm")
    expert_traj[:, 1] = 2.* ((expert_traj[:, 1] / env.height) - 0.5)
    expert_traj[:, 0] = 2.* ((expert_traj[:, 0] / env.width) - 0.5)
    expert_traj[:, 2] /= env.initial_speed
    expert_traj[:, 3] /= env.initial_speed
    if expert_traj.shape[1] > num_inputs:
        expert_traj[:, 1 + num_inputs] = 2.* ((expert_traj[:, 1 + num_inputs] / env.height) - 0.5)
        expert_traj[:, 0 + num_inputs] = 2.* ((expert_traj[:, 0 + num_inputs] / env.width) - 0.5)
        expert_traj[:, 2 + num_inputs] /= env.initial_speed
        expert_traj[:, 3 + num_inputs] /= env.initial_speed
    return expert_traj


def evaluate(env, policy_net, value_net, discriminator, episode, best_reward, save_path, args, all_trajs=None):
    env.seed(1234)
    with torch.no_grad():
        avg_reward = 0.0
        avg_step_count = 0.
        frames = []
        for epo in range(args.eval_epochs):
            state = env.reset(init_range=args.init_range)
            if all_trajs:
                sel_init_obs = select_init_obs(all_trajs)
                state = env.reset_with_obs(sel_init_obs, after_norm=False)
            for step in range(10000): # Don't infinite loop while learning
                state = torch.from_numpy(state).unsqueeze(0)
                action, _, _ = policy_net(Variable(state))
                action = action.data[0].numpy()
                if args.render and episode % 100 == 0:
                    
                    # frames.append(np.transpose(env.render(), (2, 0, 1)))
                    frames.append(np.array(state).reshape(-1)[0:2])
                next_state, reward, done, _ = env.step(action)
                avg_reward += reward
                if done:
                    avg_step_count += step
                    break
                state = next_state
        wandb.log({
            "output/Average Reward": avg_reward / args.eval_epochs,
            "output/Average steps per episode": avg_step_count / args.eval_epochs
        })
        if args.render and episode % 100 == 0:
            frames = np.array(frames)
            plt.figure()
            plt.scatter(x=frames[:, 0], y=frames[:, 1], s=1,c='black')
            plt.xlim(-1, 1)  
            plt.ylim(-1, 1)  
            plt.savefig(f'temp_episode_{episode}.png')
            wandb.log({f"video/episode{episode}": wandb.Image(f'temp_episode_{episode}.png')})
            # wandb.log({f"video/episode{episode}": wandb.Video(np.array(frames), fps=120)})
        print('Evaluation: Episode ', episode, ' Reward ', avg_reward / args.eval_epochs)
        if best_reward < avg_reward / args.eval_epochs:
            best_reward = avg_reward / args.eval_epochs
            torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(), 'rew':best_reward, 'disc':discriminator.state_dict()}, save_path)
        
        if episode % 100 == 0:
            instant_path = save_path.replace('seed', f'episode_{episode}_seed')
            torch.save({'policy':policy_net.state_dict(), 'value':value_net.state_dict(), 'rew':best_reward, 'disc':discriminator.state_dict()}, instant_path)
    return best_reward
