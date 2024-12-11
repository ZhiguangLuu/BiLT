import torch
import torch.nn as nn
import numpy as np


def mixing_ratio_scheduler(sargs, epoch):
    """
    :param sargs: parsed input arguments
    :param epoch: current epoch number
    :return:
        beta: mixing weights for driving loss
        alpha: mixing weights for cosine-similarity based loss
    """

    if sargs.loss_schedule == 'linear-decrease':
        alpha = (1 - epoch / sargs.epochs)
        beta = 1.0 - alpha
    elif sargs.loss_schedule == 'linear-increase':
        alpha = epoch / sargs.epochs
        beta = 1.0 - alpha
    elif sargs.loss_schedule == 'cosine':
        p = float(sargs.loss_schedule_period)
        alpha = 0.5*(1 - np.cos(2 * (np.pi / p) * epoch))
        beta = 1.0 - alpha
    elif sargs.loss_schedule == 'cosine-linear-increase':
        p = float(sargs.loss_schedule_period)
        alpha = 0.5 * (1 - np.cos(2 * (np.pi / p) * epoch)) * epoch / sargs.epochs
        beta = 1.0 - alpha
    elif sargs.loss_schedule == 'cosine-linear-decrease':
        p = float(sargs.loss_schedule_period)
        beta = 0.5 * (1 - np.cos(2 * (np.pi / p) * epoch)) * epoch / sargs.epochs
        alpha = 1.0 - beta
    else:
        alpha = float(sargs.loss_schedule)
        if alpha <= 1.0:
            beta = 1.0 - alpha
        else:  # if alpha > 1.0, this is relaxing constraint: beta + alpha = 1.0
            beta = 1.0

    return beta, alpha


# Squared cosine similarity
class GeneralSquareCosineSimilarityLoss(nn.Module):
    def __init__(self, num_classes, reduction='batchmean'):
        super(GeneralSquareCosineSimilarityLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, fixed_cls, x, y):
        """
        fixed_cls: tensor with shape (num_classes, num_features)
        x: tensor with shape (batch_size, num_features), raw features
        y: tensor with shape (batch_size, num_classes), simlabel
        """

        sim = []
        for c in range(self.num_classes):
            # similarity between features and classifier c
            sim.append(
                torch.unsqueeze(
                    nn.functional.cosine_similarity(fixed_cls[c], x, dim=1), dim=1
                )
            )

        # sim tensor: (batch_size, num_classes) -> mimic (example,logits) format
        sim = torch.cat(sim, dim=1)
        sq_diff = torch.square(sim - y)
        if self.reduction == 'batchmean':
            loss = torch.sum(sq_diff)
            loss = loss / y.size()[0]
        elif self.reduction == 'sum':
            loss = torch.sum(sq_diff)
        else:
            raise ValueError(f"unrecognized reduction '{self.reduction}'")
        return loss


# mixed-ce-gscsl
class MixedLoss_CEandGeneralSCSL(nn.Module):
    def __init__(self, num_classes, reduction='batchmean'):
        super(MixedLoss_CEandGeneralSCSL, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        # kl-divergence loss
        if self.reduction == 'batchmean':
            self.CE = nn.CrossEntropyLoss(reduction='mean')
            # squared cosine similarity loss
            self.GSCSL = GeneralSquareCosineSimilarityLoss(self.num_classes, reduction=self.reduction)
        elif self.reduction == 'sum':
            self.CE = nn.CrossEntropyLoss(reduction='sum')
            self.GSCSL = GeneralSquareCosineSimilarityLoss(self.num_classes, reduction=self.reduction)

    def forward(self, beta, alpha, fixed_cls, x, y_scaler, y_simlabel):
        """
        mixed_ratio: ratio between KL and SCSL
        fixed_cls: fixed classifier vectors, (num_classes, num_features)
        x: raw penultimate features, (batch_size, num_features)
        y_scaler: single labels, (batch_size, )
        y_softlabel: hie-aware soft-labels (similarities), (batch_size, num_classes)
        """
        logits = torch.matmul(x, fixed_cls.T)

        loss = beta * self.CE(logits, y_scaler) + \
               alpha * self.GSCSL(fixed_cls, x, y_simlabel)
        return loss
# EOF
