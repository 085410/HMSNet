import torch
import torch.nn as nn
from torch.nn import functional as F
from settings import config

# criterion.py中，词汇criterion为评判，在深度学习中，一般指损失函数方面的内容

# 交叉熵损失
class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]
        if len(score) == 1:
            return self._forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")
