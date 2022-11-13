import torch
from torch import nn
import math

from .utils import build_mlp, reparameterize, evaluate_lop_pi, evaluate_lop_pi_inf


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        # self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))
        self.log_stds = nn.Parameter(torch.ones(1, action_shape[0]) * math.log(0.5))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        # clip!
        return evaluate_lop_pi(self.net(states), torch.clamp(self.log_stds, min=-5), actions)
    
    def evaluate_log_pi_inf(self, states, next_states):
        # does not require that actions is between -1:1!
        return evaluate_lop_pi_inf(self.net(states), torch.clamp(self.log_stds, min=-5), next_states)


class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))
