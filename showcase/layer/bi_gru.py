import torch.nn as nn
import torch


class BiGRU(nn.Module):
    def __init__(self, dim_in: int, dim_u: int, num_layers: int, dropout: int):
        super(BiGRU, self).__init__()

        self.dim_in = dim_u
        self.dim_u = dim_u
        self.depth = num_layers

        # input is a set of embedding and predicate marker
        self.gru_in = nn.GRU(dim_in, dim_u, dropout=dropout, num_layers=1, batch_first=True)
        self.grus = nn.ModuleList([nn.GRU(dim_u, dim_u, dropout=dropout, num_layers=1, batch_first=True) for _ in
                                   range(num_layers - 1)])

    def _is_model_on_gpu(self):
        return next(self.parameters()).is_cuda

    def forward(self, x):
        out, _ = self.gru_in(x)
        for gru in self.grus:
            flipped = self.reverse(out.transpose(0, 1)).transpose(0, 1)
            output, _ = gru(flipped)
            out = flipped + output

        return self.reverse(out.transpose(0, 1)).transpose(0, 1)

    def reverse(self, x):
        idx = torch.arange(x.size(0) - 1, -1, -1).long()
        idx = torch.LongTensor(idx)
        if self._is_model_on_gpu():
            idx = idx.cuda()
        return x[idx]
