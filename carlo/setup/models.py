import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=64):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)

        self.action_mean = nn.Linear(hidden_size, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x, mean_only=False):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        if mean_only:
            return action_mean
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

class CloningPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(CloningPolicy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.affine3 = nn.Linear(hidden_size, hidden_size)

        self.action_mean = nn.Linear(hidden_size, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.affine3(x))

        action_mean = self.action_mean(x)

        return action_mean


class Value(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.linear3(x)
        return output


class Discriminator1(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Discriminator1, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.linear3(x)
        output = self.sigmoid(output)
        return output


class RNNDiscriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(RNNDiscriminator, self).__init__()
        self.sequence_model = nn.LSTM(num_inputs, hidden_size, num_layers=1, bias=True, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (x, _) = self.sequence_model(x)
        x = x.view(x.size(1), -1)
        x = F.tanh(self.linear2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.linear3(x)
        output = self.sigmoid(output)
        return output


class ForwardModel(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, num_layers=6):
        super(ForwardModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.affine_layers = nn.ModuleList()
        #self.bn_layers = nn.ModuleList()
        self.layers = num_layers
        self.first_layer = nn.Linear(self.in_features, hidden_dim)
        for i in range(self.layers):
            self.affine_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_dim, out_features)


    def forward(self, inputs):
        last_output = self.relu(self.first_layer(inputs))
        for i, affine in enumerate(self.affine_layers):
            res = self.relu(affine(last_output))
            output = self.relu(last_output+res)
            last_output = output
        delta = self.fc(last_output)
        return delta


class InverseModel(nn.Module):
    def __init__(self, num_inputs, hidden_dim, action_dim, num_layers=6):
        super(InverseModel, self).__init__()
        self.in_features = num_inputs
        self.out_features = action_dim
        self.affine_layers = nn.ModuleList()
        #self.bn_layers = nn.ModuleList()
        self.layers = num_layers
        self.first_layer = nn.Linear(self.in_features, hidden_dim)
        for i in range(self.layers):
            self.affine_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.relu = nn.ReLU()
        self.final = nn.Linear(hidden_dim, self.out_features)


    def forward(self, inputs):
        last_output = self.relu(self.first_layer(inputs))
        for i, affine in enumerate(self.affine_layers):
            res = self.relu(affine(last_output))
            output = self.relu(last_output+res)
            last_output = output
        action = self.final(last_output)
        return action


class ExpectReward(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(ExpectReward, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.linear3(x)
        return output


class StatePredictor(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(StatePredictor, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_inputs)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.linear3(x)
        return output

class Generator(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.d1(torch.tanh(self.fc1(x)))
        x = self.d2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x
