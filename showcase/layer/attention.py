import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchAttentionLayer(nn.Module):
    def __init__(self, dim_u: int, dim_target: int):
        super(BatchAttentionLayer, self).__init__()
        self.score = nn.Linear(dim_u + dim_target, 1)
        self.linear = nn.Linear(dim_u + dim_target, dim_u + dim_target)

    def forward(self, rnn_outputs, target):
        outputs = torch.cat(
            [rnn_outputs.repeat(target.size(0), 1, 1), target.repeat(rnn_outputs.size(0), 1, 1).transpose(0, 1)], dim=2)
        outputs = self.linear(outputs)
        outputs = self.score(F.tanh(outputs)).view(target.size(0), rnn_outputs.size(0))  # curr_tokens, prev_tokens
        weights = F.softmax(outputs, dim=1)
        return torch.stack([torch.sum(torch.t(rnn_outputs * w.view(w.size(0), 1)), 1) for w in weights])
