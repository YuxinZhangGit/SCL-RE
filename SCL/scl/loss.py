from abc import ABC
from ast import Lambda

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class WWWLoss(Loss):
    def __init__(self, rel_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm):
        self._rel_criterion = rel_criterion
        self._entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self.temperature = 0.07
        self.lamda = torch.rand(1,requires_grad=True).to(model.device)
        self.beta = torch.rand(1,requires_grad=True).to(model.device)

    '''     
    对比学习损失计算函数
    :param rel_log: 一个batch中关系logit变量
    :param rel_types: 一个batch中关系类型label
    :return: 返回batch的对比学习损失
    '''
    def get_sc_loss(self, rel_log: torch.Tensor, rel_types: torch.Tensor):

        mask = torch.matmul(rel_types,rel_types.T)  # 
        rel_log = rel_log.view(mask.shape[0],-1) # logits
        anchor_dot_contrast = torch.div(torch.matmul(rel_log, rel_log.T),self.temperature)  
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = -torch.div(anchor_dot_contrast,logits_max.detach())
        # logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(rel_log.shape[0]).view(-1, 1).to(mask.device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask #所有的
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        exp_logits_pos = torch.exp(logits) * mask
        # mask2 = mask.view(rel_logits.shape[0],-1).contiguous()
        # loc = torch.where(mask2 == 0)
        # mask2[loc] = 1
        a = mask.sum(1)
        loc = torch.where(a == 0)
        a[loc] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / a #sum里面有0，所以会出现inf
        loss = - (self.temperature / self.temperature) * mean_log_prob_pos
        loss = loss.view(rel_log.shape[0]).mean()
        # d = exp_logits.sum(1, keepdim=True)
        # e = torch.log(exp_logits.sum(1, keepdim=True))
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # a = torch.log(exp_logits_pos.sum(1) / exp_logits.sum(1))
        # b = exp_logits_pos.sum(1) / exp_logits.sum(1)
        # log_prob = logits - torch.log(exp_logits_pos.sum(1) / exp_logits.sum(1))

        # a = (mask * log_prob).sum(1)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # b = exp_logits.sum(1)
        # c = exp_logits_pos.sum(1)
        # mean_pos = torch.log(exp_logits_pos.sum(1) / exp_logits.sum(1))

        # loss = - (self.temperature / self.temperature) * log_prob
        # loss = loss.mean()

        return loss

    def compute(self, entity_logits, rel_logits, rel_repr, entity_types, rel_types, entity_sample_masks, rel_sample_masks):
        # 实体损失
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        # 关系分类损失
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_logits = rel_logits.view(-1, rel_logits.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_logits, rel_types)
            rel_loss = rel_loss.sum(-1) / rel_loss.shape[-1]
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            #对比学习损失计算
            rel_sc_loss = self.get_sc_loss(rel_repr,rel_types)


            # 联合模型损失
            # train_loss = entity_loss + self.lamda * rel_loss + (1-self.lamda) * rel_sc_loss
            train_loss = entity_loss + self.lamda * rel_loss + ( 1 - self.lamda ) * rel_sc_loss
        else:
            # 当前数据没有关系三元组样本时！
            train_loss = entity_loss
            # train_loss = entity_loss

        #根据损失函数直接反向传播
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
        self._optimizer.step()
        self._scheduler.step()
        self._model.zero_grad()
        return train_loss.item()