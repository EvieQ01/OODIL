import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim import Adam
import numpy as np
import math
from tqdm import trange

from .ppo import PPO
from gail_airl_ppo.network import GAILDiscrim
import os
import wandb


class GAILFO(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=50000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc=(100, 100), epoch_ppo=50, epoch_disc=10,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0, pen_lambda=0.0, use_minibatch=False, use_vectorenv=False):

        if use_minibatch:
            lr_actor = 5e-5
            lr_critic = 5e-5
            lr_disc = 5e-5
            clip_eps = 0.2
            epoch_ppo = 10
            epoch_disc = 20
            units_actor = (128, 128)
            units_critic = (128, 128)
            units_disc = (256, 256)
            lambd = 0.97
            pen_lambda = 10.0
            self.minibatch_size = 128
            dropout=None

        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = GAILDiscrim(
            state_shape=state_shape,
            action_shape=state_shape,  # action is next_state!!
            hidden_units=units_disc,
            hidden_activation=nn.Tanh(),
        ).to(device)

        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc
        self.pen_lambda = pen_lambda

        self.use_minibatch = use_minibatch
        self.use_vectorenv = use_vectorenv

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            if self.use_minibatch:
                # next_states serve as actions
                states, _, _, _, _, actions = self.buffer.get()
                # Update discriminator.
                self.update_disc_minibatch(states, actions, writer)
            else:
                # Samples from current policy's trajectories.
                # next_states serve as actions
                states, _, _, _, _, actions = self.buffer.sample(self.batch_size)
                # Samples from expert's demonstrations.
                states_exp, _, _, _, actions_exp = \
                    self.buffer_exp.sample(self.batch_size)
                # Update discriminator.
                self.update_disc(states, actions, states_exp, actions_exp, writer)

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()

        # Calculate rewards.
        rewards = self.disc.calculate_reward(states, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

    def update_disc_minibatch(self, states, actions, writer):
        batch_size = self.minibatch_size

        # random permutation
        imitator_perm = np.arange(states.shape[0])
        np.random.shuffle(imitator_perm)
        states = states[imitator_perm].clone()
        actions = actions[imitator_perm].clone()

        batch_num = int(math.ceil(states.shape[0] / batch_size))
        for b in trange(batch_num, desc='GAIL epoch'):
            ind = slice(b * batch_size, min((b + 1) * batch_size, states.shape[0]))
            states_b, actions_b = states[ind], actions[ind]

            # Samples from expert's demonstrations.
            states_exp_b, _, _, _, actions_exp_b = self.buffer_exp.sample(batch_size)

            self.update_disc(states_b, actions_b, states_exp_b, actions_exp_b, writer)

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(fake_data.shape[0], 1)
        idx = np.random.randint(0, len(real_data), fake_data.shape[0])
        real_data_b = real_data[idx]

        alpha = alpha.expand(real_data_b.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data_b + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(self.device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.disc.forward_one(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                            ** 2).mean() * self.pen_lambda
        return gradient_penalty

    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        grad_pen = self.calc_gradient_penalty(
            torch.cat([states_exp, actions_exp], dim=-1), torch.cat([states, actions], dim=-1))

        self.optim_disc.zero_grad()
        (loss_disc + grad_pen).backward()
        self.optim_disc.step()

        if self.learning_steps_disc % self.epoch_disc == 0:
            # writer.add_scalar(
            #     'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            # writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            # writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
            writer.add_scalar('debug/discriminator loss', loss_disc.item())
            writer.add_scalar('debug/acc_pi', acc_pi)
            writer.add_scalar('debug/acc_exp', acc_exp)
            
            writer.add_scalar('debug/discriminator real', logits_exp.mean())
            writer.add_scalar('debug/discriminator fake', logits_pi.mean())
            writer.add_scalar('debug/discriminator grad_pen', grad_pen.mean())

    def save_models(self, save_dir):
        super().save_models(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We  save actor, disc, critic.
        print('=> save at ', save_dir)
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'gail_actor.pth'))
        torch.save(self.disc.state_dict(), os.path.join(save_dir, 'gail_disc.pth'))
        torch.save(self.critic.state_dict(), os.path.join(save_dir, 'gail_value.pth'))
