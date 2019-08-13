import torch.nn as nn

class BCELoss2d(nn.Module):
    
    def __init__(self):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predict, target):
        predict = predict.view(-1)
        target = target.view(-1)
        return self.bce_loss(predict, target)

def dice_coeff(predict, target):
    smooth = 0.001
    batch_size = predict.size(0)
    predict = (predict > 0.5).float()
    m1 = predict.view(batch_size, -1)
    m2 = target.view(batch_size, -1)
    intersection = (m1 * m2).sum(-1)
    return ((2.0 * intersection + smooth) / (m1.sum(-1) + m2.sum(-1) + smooth)).mean()
