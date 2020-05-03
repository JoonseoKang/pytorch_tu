import torch
import torch.nn.functional as F

torch.manual_seed(1)

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
print(hypothesis)
hypothesis.sum()

z = torch.rand(3, 5, requires_grad=True)

hypothesis = F.softmax(z, dim=1)
print(hypothesis)

y = torch.randint(5, (3,)).long()

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)
print(y.unsqueeze(1))

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
print(cost)

F.nll_loss(F.log_softmax(z, dim=1), y)
F.cross_entropy(z, y)
#F.cross_entropy는 비용 함수에 소프트맥스 함수까지 포함하고 있음을 기억하고 있어야 구현 시 혼동하지 않습니다.

