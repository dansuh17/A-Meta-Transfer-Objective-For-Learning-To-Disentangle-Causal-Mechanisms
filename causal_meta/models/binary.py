import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from causal_meta.utils.torch_utils import logsumexp


class BinaryStructuralModel(nn.Module):
    def __init__(self, model_A_B, model_B_A):
        super(BinaryStructuralModel, self).__init__()
        self.model_A_B = model_A_B
        self.model_B_A = model_B_A
        self.z = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, inputs):
        return self.online_loglikelihood(self.model_A_B(inputs), self.model_B_A(inputs))

    def online_loglikelihood(self, logl_A_B, logl_B_A):
        log_alpha, log_1_m_alpha = F.logsigmoid(self.z), F.logsigmoid(-self.z)

        return logsumexp(
            log_alpha + torch.sum(logl_A_B),
            log_1_m_alpha + torch.sum(logl_B_A))

    def modules_parameters(self):
        # parameters for conditional models
        return chain(self.model_A_B.parameters(), self.model_B_A.parameters())

    def structural_parameters(self):
        # meta-parameter "z" that determines which conditional model is more probable
        return [self.z]


class ModelA2B(nn.Module):
    # models A -> B
    def __init__(self, marginal, conditional):
        super(ModelA2B, self).__init__()
        self.p_A = marginal
        self.p_B_A = conditional

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, split_size_or_sections=1, dim=1)
        # sum of log-softmax values
        return self.p_A(inputs_A) + self.p_B_A(inputs_B, inputs_A)


class ModelB2A(nn.Module):
    # models B -> A
    def __init__(self, marginal, conditional):
        super(ModelB2A, self).__init__()
        self.p_B = marginal
        self.p_A_B = conditional

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        # sum of log-softmax values
        return self.p_B(inputs_B) + self.p_A_B(inputs_A, inputs_B)
