import numpy as np
import torch
from torch import nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DRQN(nn.Module):

    def __init__(self, inp_dim, rnn_h_dim, hidden_dim, tr_count):
        super(DRQN, self).__init__()
        self.lstm = nn.LSTM(inp_dim, rnn_h_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(rnn_h_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, tr_count, bias=True)
        )

        self.rnn_hidden_dim = rnn_h_dim

        # glorot initialization
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)

    # at the beginning of each experiment, the hidden states of the lstm are initialized
    def init_hidden(self, batch_size=1):
        hidden_state = (torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device),
                        torch.zeros(1, batch_size, self.rnn_hidden_dim, device=device))

        return hidden_state

    # at the beginning of training sequence, the hidden states of the lstm are reset.
    def reset_hidden(self, hidden_state):
        hidden_state[0][:, :, :] = 0
        hidden_state[1][:, :, :] = 0
        return hidden_state[0].detach(), hidden_state[1].detach()

    # Called with either one element to determine next action, or a batch during optimization
    def forward(self, x, hidden_state):
        x = x.to(device)
        lstm_out, next_hidden_state = self.lstm(x, hidden_state)
        output = self.mlp(lstm_out)
        return output, next_hidden_state
