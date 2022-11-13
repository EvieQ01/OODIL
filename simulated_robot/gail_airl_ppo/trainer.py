import os
import pdb
from time import time, sleep
from datetime import timedelta
from tqdm import trange, tqdm
import numpy as np
import wandb
# from torch.utils.tensorboard import SummaryWriter

from .utils import Mywriter
class Trainer:

    def __init__(self, env, env_test, algo, log_dir, seed=0, num_steps=10**5,
                 eval_interval=10**3, num_eval_episodes=50, render=False, render_fix=False):
        super().__init__()

        # Env to collect samples.
        self.env = env
        # * setting seed is moved to main.py
        # self.env.seed(seed)

        # Env for evaluation.
        self.env_test = env_test
        # self.env_test.seed(2**31-seed)

        self.algo = algo
        self.log_dir = log_dir

        # Log setting.
        self.summary_dir = os.path.join(log_dir, 'summary')
        self.writer = Mywriter()
        # self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.model_dir = os.path.join(log_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Other parameters.
        self.num_steps = num_steps
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes

        # For debugging
        self.render = render
        self.render_fix = render_fix
        self.seed = seed

        # For making visualize demonstrations.
        # self.rand_list = []
        # for _ in range(20):
        #     self.env.reset()
        #     # self.rand_list.append(np.concatenate((self.env.panda.state['joint_position'], np.ones(2) * self.env.panda.init_pos[-1])) - self.env.panda.init_pos)
        #     # rand_init = np.zeros_like(self.env.panda.init_pos)
        #     self.rand_list.append(np.concatenate(((self.env.panda.state['joint_position'] - self.env.panda.init_pos[:9]), np.zeros(2))))
        # print(self.rand_list)
        
        # self.env.reset_set_init(self.rand_list[-1])
        self.start_time = time()

    def train(self):
        # Time to start training.
        # self.evaluate(0)
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()

        for step in trange(1, self.num_steps + 1):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step(self.env, state, t, step)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.update(self.writer)
                wandb.log(self.writer.metric)
            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                # if step % 1e6 == 0:
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))

        # Wait for the logging to be finished.
        sleep(10)

    def train_vector(self, num_workers, max_episode_steps=2000):
        # Time to start training.
        # self.evaluate(0)
        # Episode's timestep.
        t = np.zeros(num_workers)
        # Initialize the environment.
        state = self.env.reset()

        progress_bar = tqdm(total=self.num_steps)
        for step in range(num_workers, self.num_steps+1, num_workers):
            # Pass to the algorithm to update state and episode timestep.
            state, t = self.algo.step_vector(self.env, state, t, step, max_episode_steps=2000)

            # Update the algorithm whenever ready.
            if self.algo.is_update(step):
                self.algo.aggregate_vector_memory(num_workers)
                self.algo.update(self.writer)
                wandb.log(self.writer.metric)
            # Evaluate regularly.
            if step % self.eval_interval == 0:
                self.evaluate(step)
                # if step % 1e6 == 0:
                self.algo.save_models(
                    os.path.join(self.model_dir, f'step{step}'))
            progress_bar.update(num_workers)

        # Wait for the logging to be finished.
        sleep(10)

    def evaluate(self, step):
        mean_return = 0.0

        for epi in range(self.num_eval_episodes):
            if self.render and epi == 0 and step % 1e6 == 0:
                frames = []
            state = self.env_test.reset()
            episode_return = 0.0
            done = False

            t = 0 # timestep
            while (not done):
                if self.render and epi == 0 and t % 20 == 0 and step % 1e6 == 0:
                    # only render the first one
                    frames.append(np.transpose(self.env_test.render(mode='rgb_array'), (2, 0, 1)))
                action = self.algo.exploit(state)
                state, reward, done, _ = self.env_test.step(action)
                episode_return += reward
                t += 1
            if self.render and epi == 0 and step % 1e6 == 0:
                wandb.log({f"video/{step//1e6}(1e6) step example_No.{epi}": wandb.Video(np.array(frames), fps=1200)})

            mean_return += episode_return / self.num_eval_episodes

        # self.writer.add_scalar('return/test', mean_return, step)
        wandb.log({'output/Average return': mean_return})

        print(f'Num steps: {step:<6}   '
              f'Return: {mean_return:<5.1f}   '
              f'Time: {self.time}')

        # render specific
        mean_return = 0.0
        epi = 0
        '''
        used for generating video with robot initialized at a fixed frame
        '''
        if self.render_fix:
            # set_init_joint_rand = np.zeros(len(self.env.panda.init_posi))
            # rand_list = []
            for set_init_joint_rand in self.rand_list:
                if step % 1e6 == 0:
                    frames = []
                self.env_test.reset()
                state = self.env_test.reset_set_init(set_init_joint_rand)
                episode_return = 0.0
                done = False

                t = 0 # timestep
                while (not done):
                    if t % 20 == 0 and step % 1e6 == 0:
                        # only render the first one
                        frames.append(np.transpose(self.env_test.render(mode='rgb_array'), (2, 0, 1)))
                    action = self.algo.exploit(state)
                    state, reward, done, _ = self.env_test.step(action)
                    episode_return += reward
                    t += 1
                if step % 1e6 == 0:
                    wandb.log({f"video_fix_init/{step//1e6}(1e6) step example_No.{epi}": wandb.Video(np.array(frames), fps=1200)})
                epi += 1
                mean_return += episode_return / len(self.rand_list)

            # self.writer.add_scalar('return/test', mean_return, step)
            wandb.log({'output/Average return fix_init': mean_return})

            print(f'Num steps fix_init: {step:<6}   \n'
                f'Return fix_init: {mean_return:<5.1f}  \n '
                f'Time fix_init: {self.time}')


    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
