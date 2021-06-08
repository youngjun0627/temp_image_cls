import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class MyLoss1(nn.CrossEntropyLoss):
    def __init__(self, weights=None):
        super(MyLoss1, self).__init__()
        self.celoss = torch.nn.CrossEntropyLoss(weight=weights)
    def forward(self, output, label):
        label = label.long()
        return self.celoss(output, label)



class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.15):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                            .fill_(smoothing / (n_classes - 1)) \
                            .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets
    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
            loss = -(targets * lsm).sum(-1)
        
        if self.reduction == 'sum':
            loss = loss.sum()

        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                            .fill_(smoothing /(n_classes-1)) \
                            .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
                if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


class MyLoss2(LabelSmoothCrossEntropyLoss):
    def __init__(self, weights = None):
        super(MyLoss2, self).__init__()
        self.weights = weights

    def forward(self, output, label):
        celoss = LabelSmoothCrossEntropyLoss(weight = self.weights, reduction='mean')
        loss = celoss(output, label)
        return loss

if __name__ == '__main__':
    nSamples = [887, 6130, 480, 317, 972, 101, 128]
    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    normedWeights = torch.FloatTensor(normedWeights)
    print(normedWeights)
