import torch
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min)*r

        at = torch.cat([x[i][lb] for i, lb in enumerate(target)])
        at = at.view(b, 1).repeat(1, E)

        zeros = Variable(torch.cuda.FloatTensor(x.shape).fill_(0))
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        return loss
