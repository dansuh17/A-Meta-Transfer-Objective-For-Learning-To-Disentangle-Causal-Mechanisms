import torch
import torch.nn as nn


class Marginal(nn.Module):
    def __init__(self, N, dtype=None):
        super().__init__()
        self.N = N
        # marginal probabilities; w[i] = P(A = i)
        self.w = nn.Parameter(torch.zeros(N, dtype=dtype))

    def __str__(self):
        return f'N = {self.N}\nw = {self.w}'

    def forward(self, inputs):
        # log p(A) or log p(B)
        return self.logsoftmax(inputs)

    def logsoftmax(self, inputs):
        denom = torch.logsumexp(self.w, dim=0)
        # log(exp(p(input)) / sum_i(exp(p(x_i)))) = log-softmax
        return self.w[inputs.squeeze(dim=1)] - denom


class Conditional(nn.Module):
    def __init__(self, N, dtype=None):
        super().__init__()
        self.N = N
        # conditional probabilities; w[i][j] = P(B = j | A = i)
        self.w = nn.Parameter(torch.zeros((N, N), dtype=dtype))

    def __str__(self):
        return f'N = {self.N}\nw = {self.w}'

    def forward(self, inputs, conds):
        return self.conditional_logsoftmax(inputs, conditional_values=conds)

    def conditional_logsoftmax(self, inputs, conditional_values):
        # log p(B | A) or log p(A | B)
        conds_ = conditional_values.squeeze(1)
        denoms = torch.logsumexp(self.w[conds_], dim=1)
        # log-softmax
        return self.w[conds_, inputs.squeeze(1)] - denoms


if __name__ == '__main__':
    m = Marginal(10)
    dummy_input = torch.LongTensor(10, 1).random_(10)
    print(m)
    print(m(dummy_input))
