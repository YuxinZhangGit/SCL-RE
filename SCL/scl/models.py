import torch
from torch import batch_norm, nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
import torch.nn.functional as F

from SCL import sampling
from SCL import util


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ 获取 [CLS] 编码向量 """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

def get_rel_emb(h: torch.tensor, rel_pot_embedding: torch.tensor):
    batch_size = h.shape[0]
    #获取CLS表示batch_size * dim * 1
    cls_embedding = h[:,0,:].view(batch_size,h.shape[-1],-1)
    #获取关系表示batch_size * rel_type * dim
    rel_emb = rel_pot_embedding.unsqueeze(0).repeat(batch_size,1,1)
    attention = torch.bmm(rel_emb,cls_embedding).transpose(-1,-2)
    attention = F.softmax(attention,dim=-1)
    rel_h = torch.bmm(attention,rel_emb)
    return torch.squeeze(rel_h,axis = -2)


class SCL(BertPreTrainedModel):

    VERSION = '1.1'

    def __init__(self, config: BertConfig, cls_token: int, relation_types: int, entity_types: int,
                 size_embedding: int, prop_drop: float, freeze_transformer: bool, max_pairs: int = 100):
        super(SCL, self).__init__(config)

        # 加载预训练模型
        self.bert = BertModel(config)

        # 定义模型层 或 参数
        self.rel_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, relation_types)
        self.asso_classifier = nn.Linear(config.hidden_size * 3 + size_embedding * 2, 2)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, size_embedding)
        #=================================添加关系潜在变量=================================#
        self.rel_pot_embedding = nn.Parameter(torch.Tensor(relation_types,config.hidden_size))
        nn.init.normal_(self.rel_pot_embedding,mean=0,std=1)

        self.dropout = nn.Dropout(prop_drop)

        self._cls_token = cls_token
        self._relation_types = relation_types
        self._entity_types = entity_types
        self._max_pairs = max_pairs

        self.init_weights()
        # 冻结预训练模型参数
        if freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

    # 训练前向流程
    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, relations: torch.tensor, rel_masks: torch.tensor):
        # get contextualized token embeddings
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]

        # classify spans -- 过滤实体
        size_embeddings = self.size_embeddings(entity_sizes)  # embed span candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # classify assosiation -- 分类关联
        if not entity_spans_pool.shape[0]:
            span_clf, spans_assosiation = self._classify_spans(entity_spans_pool, size_embeddings, relations, rel_masks, h_large, i) 

        # classify relations -- 关系分类器
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)
        for i in range(0, relations.shape[1], self._max_pairs):
            chunk_rel_logits,rel_repr = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_logits
        # temp = span_clf
        # temp = spans_assosiation
        return entity_clf, rel_clf ,rel_repr

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                           entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        batch_size = encodings.shape[0]
        ctx_size = context_masks.shape[-1]

        # classify entities
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings)

        # ignore entity candidates that do not constitute an actual entity for relations (based on classifier)
        relations, rel_masks, rel_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)

        rel_sample_masks = rel_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(relations.shape[1], self._max_pairs), 1), 1, 1)
        rel_clf = torch.zeros([batch_size, relations.shape[1], self._relation_types]).to(
            self.rel_classifier.weight.device)

        # obtain relation logits
        # chunk processing to reduce memory usage
        for i in range(0, relations.shape[1], self._max_pairs):
            # classify relation candidates
            chunk_rel_logits,_ = self._classify_relations(entity_spans_pool, size_embeddings,
                                                        relations, rel_masks, h_large, i)
            # apply sigmoid
            chunk_rel_clf = torch.sigmoid(chunk_rel_logits)
            rel_clf[:, i:i + self._max_pairs, :] = chunk_rel_clf

        rel_clf = rel_clf * rel_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, rel_clf, relations

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings):
        # max pool entity candidate spans
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]

        # get cls token as candidate context representation
        #======================================将cls表示替换为潜在关系表示=====================================#
        entity_ctx = get_rel_emb(h, self.rel_pot_embedding)
        # entity_ctx = get_token(h,encodings,self._cls_token)

        # 构建实体表示（拼接最大池化token+size嵌入）
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # 实体分类
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_relations(self, entity_spans, size_embeddings, relations, rel_masks, h, chunk_start):
        batch_size = relations.shape[0]

        # 结块操作，就是把实体对打包成一块块的数据，防止爆显存
        if relations.shape[1] > self._max_pairs:
            relations = relations[:, chunk_start:chunk_start + self._max_pairs]
            rel_masks = rel_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :relations.shape[1], :]

        # 获取候选实体对
        entity_pairs = util.batch_index(entity_spans, relations)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # 获取size潜入
        size_pair_embeddings = util.batch_index(size_embeddings, relations)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # 获取上下文表示
        m = ((rel_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        rel_ctx = rel_ctx.max(dim=2)[0]
        rel_ctx[rel_masks.to(torch.uint8).any(-1) == 0] = 0

        # 创建关系三元组的表示向量（通过拼接）
        rel_repr = torch.cat([rel_ctx, entity_pairs, size_pair_embeddings], dim=2)
        rel_repr = self.dropout(rel_repr)

        # 关系分类
        chunk_rel_logits = self.rel_classifier(rel_repr)
        return chunk_rel_logits,rel_repr
 
    def _classify_spans(self, entity_spans, size_embeddings, assosiation, asso_masks, h, chunk_start):
        batch_size = assosiation.shape[0]

        span_pairs = util.batch_index(entity_spans, assosiation)
        span_pairs = span_pairs.view(batch_size, span_pairs.shape[1], -1)

        size_pair_embeddings = util.batch_index(size_embeddings, assosiation)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        m = ((asso_masks == 0).float() * (-1e30)).unsqueeze(-1)
        asso_ctx = m + h
        # 最大池化 获取 上下文表示
        asso_ctx = asso_masks.max(dim=2)[0]
        asso_ctx[asso_masks.to(torch.uint8).any(-1) == 0] = 0

        # 创建一个和头尾实体表示相关的span表示向量，其实就是简单的拼接（实体表示+size表示+关系潜在表示）
        asso_repr = torch.cat([asso_masks, span_pairs, size_pair_embeddings], dim=2)
        asso_repr = self.dropout(asso_repr)

        # span分类
        chunk_acco_logits = self.asso_classifier(asso_repr)
        return asso_repr,chunk_acco_logits

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_relations = []
        batch_rel_masks = []
        batch_rel_sample_masks = []

        for i in range(batch_size):
            rels = []
            rel_masks = []
            sample_masks = []

            # get spans classified as entities
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create relations and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        rel_masks.append(sampling.create_rel_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_relations.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_rel_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_rel_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_relations.append(torch.tensor(rels, dtype=torch.long))
                batch_rel_masks.append(torch.stack(rel_masks))
                batch_rel_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.rel_classifier.weight.device
        batch_relations = util.padded_stack(batch_relations).to(device)
        batch_rel_masks = util.padded_stack(batch_rel_masks).to(device)
        batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(device)

        return batch_relations, batch_rel_masks, batch_rel_sample_masks

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


# Model access

_MODELS = {
    'SCL': SCL,
}


def get_model(name):
    return _MODELS[name]
