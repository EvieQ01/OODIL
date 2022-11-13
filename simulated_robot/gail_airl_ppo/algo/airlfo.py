
import os
import pdb
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPO
from gail_airl_ppo.network import AIRLDiscrim, disc
from gail_airl_ppo.network import StateIndependentPolicy

class AIRLFO(PPO):

    def __init__(self, buffer_exp, state_shape, action_shape, device, seed,
                 gamma=0.995, rollout_length=10000, mix_buffer=1,
                 batch_size=64, lr_actor=3e-4, lr_critic=3e-4, lr_disc=3e-4,
                 units_actor=(64, 64), units_critic=(64, 64),
                 units_disc_r=(100, 100), units_disc_v=(100, 100),
                 epoch_ppo=50, epoch_disc=10, clip_eps=0.2, lambd=0.97,
                 coef_ent=0.0, max_grad_norm=10.0):
        super().__init__(
            state_shape, action_shape, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # Discriminator.
        self.disc = AIRLDiscrim(
            state_shape=state_shape,
            gamma=gamma,
            hidden_units_r=units_disc_r,
            hidden_units_v=units_disc_v,
            hidden_activation_r=nn.ReLU(inplace=True),
            hidden_activation_v=nn.ReLU(inplace=True)
        ).to(device)

        # an extra Dynamic Model: to evaluate transition s-> s'
        # same as PPO.Actor
        self.dynamic_model = StateIndependentPolicy(
            state_shape=state_shape,
            action_shape=state_shape,
            hidden_units=units_actor,
            hidden_activation=nn.Tanh()
        ).to(device)
        self.optim_dynamic_model = Adam(self.dynamic_model.parameters(), lr=lr_actor)

        # others
        self.learning_steps_disc = 0
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, _, _, dones, log_pis, next_states = \
                self.buffer.sample(self.batch_size)
            # Samples from expert's demonstrations.
            states_exp, actions_exp, _, dones_exp, next_states_exp = \
                self.buffer_exp.sample(self.batch_size)
            
            # Calculate log probabilities of expert actions.
            # change into p(s'| s) instead of p(a|s) !!
            with torch.no_grad():
                log_pis_exp = self.dynamic_model.evaluate_log_pi(
                    states_exp, next_states_exp)
            # with torch.no_grad():
            #     log_pis_exp = self.actor.evaluate_log_pi(
            #         states_exp, actions_exp)

            # Update discriminator.
            self.update_disc(
                states, dones, log_pis, next_states, states_exp,
                dones_exp, log_pis_exp, next_states_exp, writer
            )

        # We don't use reward signals here,
        states, actions, _, dones, log_pis, next_states = self.buffer.get()
        
        # Calculate rewards.
        rewards = self.disc.calculate_reward(
            states, dones, log_pis, next_states)

        # Update PPO using estimated rewards.
        self.update_ppo(
            states, actions, rewards, dones, log_pis, next_states, writer)

        # Update the model to evaluate p(s' | s)
        self.update_dynamic_model(states, next_states, writer)

    def update_dynamic_model(self, states, next_states, writer):
        '''
        Same as update_ppo.update_actor
        '''
        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            log_pis = self.dynamic_model.evaluate_log_pi_inf(states, next_states)
            model_loss = -log_pis.mean()

            
            self.optim_dynamic_model.zero_grad()
            model_loss.backward(retain_graph=False)
            # nn.utils.clip_grad_norm_(self.dynamic_model.parameters(), self.max_grad_norm)
            print('debug/dynamic.log_std', self.dynamic_model.log_stds.mean())
            self.optim_dynamic_model.step()

            if self.learning_steps_ppo % self.epoch_ppo == 0:
                
                writer.add_scalar(
                    'debug/dynamic -loglikelihood', -log_pis.mean()) # need to sum at last dim?
                writer.add_scalar(
                    'debug/dynamic.log_std', self.dynamic_model.log_stds.mean()) # need to sum at last dim?
                print('debug/dynamic -loglikelihood', -log_pis.mean())
                print('debug/dynamic.log_std', self.dynamic_model.log_stds.mean())
                # writer.add_scalar(
                #     'debug/entropy', entropy.item())
        # Just borrow the update frequency from ppo
        # change it back
        self.learning_steps_ppo -= self.epoch_ppo


    def update_disc(self, states, dones, log_pis, next_states,
                    states_exp, dones_exp, log_pis_exp,
                    next_states_exp, writer=None):
        # Output of discriminator is (-inf, inf), not [0, 1].
        logits_pi = self.disc(states, dones, log_pis, next_states)
        logits_exp = self.disc(
            states_exp, dones_exp, log_pis_exp, next_states_exp)

        # Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = loss_pi + loss_exp

        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()
        info = {}
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
            writer.add_scalar('debug/acc_pi',acc_pi)
            writer.add_scalar('debug/acc_exp',acc_exp)
            
            writer.add_scalar('debug/discriminator real', logits_exp.mean())
            writer.add_scalar('debug/discriminator fake', logits_pi.mean())
        #     info['debug/discriminator loss'] = loss_disc.item()
        #     info['debug/acc_pi'] = acc_pi
        #     info['debug/acc_exp'] = acc_exp
        #     info['debug/discriminator real'] = logits_exp
        #     info['debug/discriminator fake'] = logits_pi
        # if writer is not None:
        #     return info
        # return None
    def step(self, env, state, t, step):
        t += 1

        action, log_pi_action_base = self.explore(state)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        # if t == env._max_episode_steps - 1:
        
        # log_pi = self.dynamic_model.evaluate_log_pi(states=torch.from_numpy(state).to(torch.float32), actions=torch.from_numpy(next_state).to(torch.float32))
        
        with torch.no_grad():
            _, log_pi = self.dynamic_model.sample(torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0))

        self.buffer.append(state, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()

        return next_state, t
    def save_models(self, save_dir):        
        super().save_models(save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We only save actor to reduce workloads.
        torch.save(self.actor.state_dict(),os.path.join(save_dir, 'airl_actor.pth'))
        torch.save(self.disc.state_dict(),os.path.join(save_dir, 'airl_disc.pth'))
        torch.save(self.critic.state_dict(),os.path.join(save_dir, 'airl_value.pth'))
