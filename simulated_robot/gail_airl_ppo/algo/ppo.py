import pdb
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
import math
from tqdm import trange, tqdm

from .base import Algorithm
from gail_airl_ppo.buffer import RolloutBuffer
from gail_airl_ppo.network import StateIndependentPolicy, StateFunction


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


class PPO(Algorithm):

    def __init__(self, state_shape, action_shape, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=20, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=(64, 64), units_critic=(64, 64),
                 epoch_ppo=10, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0, value_clip_epsilon=10.0, value_iter=2):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # Rollout buffer.
        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=device,
            mix=mix_buffer
        )

        # Actor.
        self.actor = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=action_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)

        # Critic.
        self.critic = StateFunction(
            state_shape=state_shape,
            hidden_units=units_critic,
            hidden_activation=nn.Tanh()
        ).to(device)

        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.value_clip_epsilon = value_clip_epsilon
        self.value_iter = value_iter
        self.device = device

        self.use_minibatch = False

    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        # if t == env._max_episode_steps - 1:
        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t

    def step_vector(self, env, state, t, step, max_episode_steps=2000):
        t += 1

        action, log_pi = self.explore_vector(state)
        next_state, reward, done, _ = env.step(action)
        # mask = False if t == env._max_episode_steps else done
        mask = (t == max_episode_steps).astype(np.float)
        mask = (np.zeros(mask.shape) * mask + done * (1-mask)).astype(np.bool)

        if not hasattr(self, 'vector_memory'):
            self.vector_memory = []
        self.vector_memory.append((state, action, reward, mask, log_pi, next_state))
        # self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done.any():
            t = 0 * done.astype(np.float) + t * (1-done.astype(np.float))

        return next_state, t

    def aggregate_vector_memory(self, num_workers):
        items = [], [], [], [], [], []
        for x in tqdm(self.vector_memory):
            for k in range(len(x)):
                items[k].append(x[k])
        items = [np.array(x) for x in items]
        for i in range(num_workers):
            curr_items = [x[:,i] for x in items]
            self.buffer.append_vector(*curr_items)
        self.vector_memory = []

    def update(self, writer):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = \
            self.buffer.get()
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_ppo(self, states, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            if self.use_minibatch:
                self.update_actor_critic_minibatch(states, actions, log_pis, gaes, targets, writer)
            else:
                self.update_critic(states, targets, writer)
                self.update_actor(states, actions, log_pis, gaes, writer)

    def update_critic(self, states, targets, writer):
        q_now = self.critic(states)
        loss_critic = (q_now - targets).pow(2)
        loss_critic = torch.clamp(loss_critic, 0, self.value_clip_epsilon)
        loss_critic = loss_critic.mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar('debug/critic loss', loss_critic.item())
            writer.add_scalar('debug/Q value', q_now.mean())
            writer.add_scalar('debug/Q target', targets.mean())
            # writer.add_scalar(
            #     'loss/critic', loss_critic.item(), self.learning_steps)

    def update_actor(self, states, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'debug/actor loss', loss_actor.item())
            writer.add_scalar(
                'debug/entropy', entropy.item())

    def update_actor_critic_minibatch(self, states, actions, log_pis, gaes, targets, writer):
        batch_size = self.minibatch_size

        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(self.device)

        states, actions, log_pis, gaes, targets = \
            states[perm].clone(), actions[perm].clone(), log_pis[perm].clone(), \
            gaes[perm].clone(), targets[perm].clone()

        optim_iter_num = int(math.ceil(states.shape[0] / batch_size))
        for i in trange(optim_iter_num, desc='PPO epoch'):
            ind = slice(i * batch_size, min((i + 1) * batch_size, states.shape[0]))
            states_b, actions_b, log_pis_b, gaes_b, targets_b = \
                states[ind], actions[ind], log_pis[ind], gaes[ind], targets[ind]
            
            for _ in range(self.value_iter):
                self.update_critic(states_b, targets_b, writer)
            self.update_actor(states_b, actions_b, log_pis_b, gaes_b, writer)

    def save_models(self, save_dir):
        pass
