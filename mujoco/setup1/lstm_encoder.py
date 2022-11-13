import torch
import torch.nn as nn
import torch.nn.functional as F


class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        '''
        param:
            input_size:the feature dim of input X
            hidden_size:the feature dim of hidden state h
            num_layers:the num of stacked lstm layers
        '''
        super(lstm_encoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

    def forward(self, x):
        '''
        param:
            x:input of lstm unit,(seq_len,batch,feature_dim)
        output:
            seq_out:give all the hidden state in the sequence, return last time-step's output, (h_n, c_n)
            last:the last h & c
        '''
        self.seq_out, (self.last_h, self.last_c) = self.lstm(x.view(x.shape[0] , x.shape[1], self.input_size))
        # return last time-step's output, (h_n, c_n)
        return self.seq_out[-1, :, :], (self.last_h, self.last_c)

