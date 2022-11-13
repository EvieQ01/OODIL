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

class lstm_decoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=1):
        '''
        param:
            input_size:the feature dim of input x
            hidden_size:the dim of hidden state h
            num_layers:the stacked num of lstm units
        '''
        super(lstm_decoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)
        self.linear=nn.Linear(hidden_size,input_size)

    def forward(self,x,encoder_hidden_state):   #这里的输入实际上只需要encoder的最后一个输出，而将x输入的原因是使用
        '''                                       teacher force 模式训练模型时需要目标作为输入
        param:
            x:should be 2D   (batch_size,input_size)
            encoder_hidden_state:the output of encoder
        output:
            de_seq_out: all the hidden state in the sequence
            de_last:the last h &c
        '''
        self.de_seq_out,self.de_last=self.lstm(x.unsqueeze(0),encoder_hidden_state)
        self.de_seq_out=self.linear(self.de_seq_out.squeeze(0))
        return self.de_seq_out,self.de_last
