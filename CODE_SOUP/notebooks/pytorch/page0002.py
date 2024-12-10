import torch
import torch.nn.functional as F

logit = torch.randn(3, 5)
prob1 = F.softmax(logit, dim=1)
prob2 = torch.nn.Softmax(dim=1)(logit)
log_prob0 = F.log_softmax(logit, dim=1)

log_prob1 = torch.log(prob1)
log_prob2 = torch.log(prob2)
log_prob3 = torch.nn.LogSoftmax(dim=1)(logit)


target = torch.randint(0, 5, (3,))
loss0 = F.nll_loss(log_prob0, target)
loss1 = F.nll_loss(log_prob1, target)
loss2 = F.nll_loss(log_prob2, target)
loss4 = F.nll_loss(log_prob3, target)
print('done')
