from torch import nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, gamma=1., reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.gamma = gamma

    def reduce_loss(self, loss, target_weights=None):
        if target_weights is not None:
            weights = (target_weights * (-self.gamma)).softmax(-1)
            return (weights * loss).sum(-1)
        else:
            return loss.mean()

    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    def forward(self, preds, target, matrix):
        target_weights = matrix[target]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds, target_weights, )
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss, nll, self.epsilon).mean()
    
