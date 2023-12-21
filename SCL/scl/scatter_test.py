import torch

# logits = torch.Tensor([[1,2,3],[1,0,3],[1,2,3]])
# # a = torch.ones_like(mask)
# # b = torch.arange(3).view(-1, 1)
# logits_mask = torch.scatter(
#             torch.ones_like(logits),
#             1,
#             torch.arange(3).view(-1, 1),
#             0
#         )
# a = torch.exp(logits)
# exp_logits = torch.exp(logits) * logits_mask
# b = torch.log(exp_logits.sum(1, keepdim=True))
# c = exp_logits.sum(1, keepdim=True)
# log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
type = torch.Tensor([[0,1,0],[1,0,0],[0,0,1],[0,1,0]])
a = type.sum(1)
type = torch.matmul(type,type.T)
print(type.shape())