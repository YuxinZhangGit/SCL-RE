from cProfile import label
from lib2to3.pgen2 import token
from typing import *
from numpy import dtype

import torch
from torch import Tensor, layer_norm, sigmoid
import torch.nn.functional as F
from torch.nn import Dropout, Linear, CrossEntropyLoss, LayerNorm, BCELoss
from transformers.models.bert.modeling_bert import BertModel, BertForMaskedLM
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM, RobertaModel
from transformers import AutoModel, AutoConfig, AutoModelForMaskedLM, BertConfig, BertTokenizer
from transformers.file_utils import PaddingStrategy
from sklearn.metrics import recall_score, f1_score, precision_score
import Levenshtein
import difflib


class ClassificationBaselineModel(torch.nn.Module):
    def __init__(self, mlm_name_or_path: str, event2id: Dict[str, int]) -> None:
        super(ClassificationBaselineModel, self).__init__()
        self.mlm: BertModel = AutoModel.from_pretrained(mlm_name_or_path)
        self.fc = Linear(self.mlm.config.hidden_size, len(event2id))
        self.dropout: Dropout = Dropout()

    def forward(self, input_ids: Tensor, masks: Tensor, token_type_ids: Tensor) -> Tensor:
        mlm_output: Tensor = self.mlm(
            input_ids, attention_mask=masks, token_type_ids=token_type_ids)
        cls: Tensor = mlm_output[0][:, 0, :]
        logits = self.fc(cls)
        return logits

class LSDRModel(torch.nn.Module):
    def __init__(self,mlm_name_or_path:str,rel2id:Dict[str,int]) -> None:
        super().__init__()
        self.mlm: BertModel = AutoModel.from_pretrained(mlm_name_or_path)
        self.rel2id = rel2id
        self.rel_type_num = len(rel2id)

        self.dropout = Dropout()
        self.fc = torch.nn.Linear(self.mlm.config.hidden_size*3,self.rel_type_num + 1)

    def forward(self,input_ids: Tensor,masks: Tensor,ent_h_ids: Tensor,ent_h_mask: Tensor,ent_t_ids: Tensor,ent_t_mask: Tensor):
        # batch_size, _ = input_ids.shape
        text_mlm_output = self.mlm(input_ids,attention_mask=masks)
        text_cls = text_mlm_output.last_hidden_state[:,0,:]
        ent_h_mlm_output = self.mlm(ent_h_ids,attention_mask=ent_h_mask)
        ent_h_cls = ent_h_mlm_output.last_hidden_state[:,0,:]
        ent_t_mlm_output = self.mlm(ent_t_ids,attention_mask=ent_t_mask)
        ent_t_cls = ent_t_mlm_output.last_hidden_state[:,0,:]

        triple_embedding = torch.cat([ent_h_cls,text_cls,ent_t_cls],dim=1)
        res_logits = self.fc(triple_embedding)

        return res_logits

def batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):
    model=trainer.model
    if isinstance(model,torch.nn.parallel.DataParallel):
        model=model.module
    elif isinstance(model,ClassificationBaselineModel):
        input_ids, masks, token_type_ids, labels = batch_data
        input_ids, masks, token_type_ids, labels =\
            input_ids.cuda(trainer.device, non_blocking=True),\
            masks.cuda(trainer.device, non_blocking=True),\
            token_type_ids.cuda(trainer.device, non_blocking=True),\
            labels.cuda(trainer.device, non_blocking=True)
        logits = trainer.model(input_ids, masks, token_type_ids)
        return labels, logits
    elif isinstance(model,LSDRModel):
        input_ids,masks,ent_h_ids,ent_h_masks,ent_t_ids,ent_t_masks,labels=batch_data
        input_ids,masks,ent_h_ids,ent_h_masks,ent_t_ids,ent_t_masks,labels=\
            input_ids.cuda(trainer.device, non_blocking=True),\
            masks.cuda(trainer.device, non_blocking=True),\
            ent_h_ids.cuda(trainer.device, non_blocking=True),\
            ent_h_masks.cuda(trainer.device, non_blocking=True),\
            ent_t_ids.cuda(trainer.device, non_blocking=True),\
            ent_t_masks.cuda(trainer.device, non_blocking=True),\
            labels.cuda(trainer.device, non_blocking=True)
        logits = trainer.model(input_ids,masks,ent_h_ids,ent_h_masks,ent_t_ids,ent_t_masks)
        return labels, logits

def batch_cal_loss_func(labels: torch.Tensor, preds: Tensor, trainer) -> Tensor:
    label = labels.reshape([-1])
    pred = preds
    loss = F.cross_entropy(pred, label,label_smoothing=0.05)
    return loss

class NewPromptModel(torch.nn.Module):
    def __init__(self, mlm_name_or_path: str) -> None:
        super(NewPromptModel, self).__init__()
        mlm_for_maskedlm: Union[BertForMaskedLM, RobertaForMaskedLM] = BertForMaskedLM.from_pretrained(
            mlm_name_or_path)
        self.mlm_config: Union[BertConfig, RobertaConfig] = AutoConfig.from_pretrained(
            mlm_name_or_path)
        self.hidden_dim = self.mlm_config.hidden_size
        if hasattr(mlm_for_maskedlm, "bert"):
            assert isinstance(mlm_for_maskedlm, BertForMaskedLM)
            self.mlm_type = "bert"
            self.mlm: BertModel = mlm_for_maskedlm.bert
            self.lm_head = mlm_for_maskedlm.cls.predictions.transform
            self.lm_decoder = mlm_for_maskedlm.cls.predictions.decoder

        elif hasattr(mlm_for_maskedlm, "roberta"):
            assert isinstance(mlm_for_maskedlm, RobertaForMaskedLM)
            self.mlm_type = "roberta"
            self.mlm: RobertaModel = mlm_for_maskedlm.roberta
            self.lm_head = torch.nn.Sequential(
                mlm_for_maskedlm.lm_head.dense,
                torch.nn.GELU(),
                mlm_for_maskedlm.lm_head.layer_norm
            )
            self.lm_decoder = mlm_for_maskedlm.lm_head.decoder
        else:
            raise NotImplemented("目前仅支持bert,roberta")

    def forward(self, input_ids: Tensor, masks: Tensor):
        # batch_size, sequence_size = input_ids.shape
        #获得输出向量
        sequence_mlm_output = self.mlm(input_ids, attention_mask=masks)
        
        sequence_embeddings: Tensor = sequence_mlm_output[0]
        lm_output: Tensor = self.lm_head(sequence_embeddings)
        # batch_size * sequence_length * 20000+
        lm_decoded: Tensor = self.lm_decoder(lm_output)
        return lm_decoded

def metrics_cal_func(metrics: Dict[str, torch.Tensor]):
    if "label_precision" in metrics:
        labels_precision = metrics['label_precision']
        preds_precision = metrics['pred_precision']
    else:
        labels_precision = []
        preds_precision = []
    if 'label_recall' in metrics:
        labels_recall = metrics['label_recall']
        preds_recall = metrics['pred_recall']
    else:
        labels_recall = []
        preds_recall = []

    precision_micro = 0.0
    recall_micro = 0.0
    f1_micro = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    f1_macro = 0.0

    precision_micro = precision_score(labels_precision, preds_precision, average="micro", zero_division=0)
    recall_micro = recall_score(labels_recall, preds_recall, average="micro", zero_division=0)

    if len(labels_precision) != 0:
        precision_macro = precision_score(labels_precision, preds_precision, average="macro", zero_division=0)
    if len(labels_recall) != 0:
        recall_macro = recall_score(labels_recall, preds_recall, average="macro", zero_division=0)

    f1_micro = 2 * precision_micro * recall_micro / ( precision_micro + recall_micro + 1e-9)
    f1_macro = 2 * precision_macro * recall_macro / ( precision_macro + recall_macro + 1e-9)

    res = {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }
    return res

def new_prompt_batch_forward_func(batch_data: Tuple[torch.Tensor, ...], trainer):
    batch_data = tuple(
        list(map(lambda x: x.cuda(trainer.device, non_blocking=True), list(batch_data))))
    input_ids, masks, masks4masks, mask_start_end, vocab_masks, input_labels, rel_types= batch_data
    decoder_output = trainer.model(input_ids, masks)
    return (masks4masks, mask_start_end, vocab_masks, input_labels, rel_types, masks), decoder_output

def new_prompt_batch_cal_loss_func(labels:  Tuple[torch.Tensor, ...], decoder_output: Tensor, trainer):
    batch_size, sequence_length, vocab_size = decoder_output.shape
    masks4masks, mask_start_end, vocab_masks, input_labels, rel_types, masks = labels
    vocab_masks_repeated = vocab_masks.reshape(
        [1, 1, -1]).repeat([batch_size, sequence_length, 1])
    decoder_output = decoder_output+vocab_masks_repeated
    loss = F.cross_entropy(
        decoder_output.reshape([-1, vocab_size]),
        input_labels.reshape([-1]),
        reduction="none"
    )
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss) | torch.isneginf(
        loss), torch.tensor(0, dtype=torch.float, device=loss.device), loss)
    loss = loss.reshape([batch_size, -1])
    loss = (loss*masks4masks.float())
    #======================================================================================================#
    loss=loss.sum(-1)
    res = loss.sum()/batch_size
    return res

class NewPromptBatchMetricsFunc(object):
    def __init__(self, mlm_name_or_path, rel2ids_zh) -> None:
        self.rel2ids_zh = rel2ids_zh
        self.rel_type_num = len(rel2ids_zh)
        self.id2rel_zh = {}
        self.rel_types = []
        for k,v in self.rel2ids_zh.items():
            self.id2rel_zh[v] = k
            self.rel_types.append(k)
        self.tokenizer = BertTokenizer.from_pretrained(mlm_name_or_path)

    def _get_event_type(self, s: str):
        t = torch.zeros([self.rel_type_num], dtype=torch.float)
        for i in range(self.rel_type_num):
            t[i] = Levenshtein.jaro(s, self.rel_types[i])
        return t.argmax().item()

    def __call__(self, labels: Tuple[torch.Tensor, ...], decoder_output: Tensor, metrics: Dict[str, List[int]], trainer) -> Any:
        masks4masks, mask_start_end, vocab_masks, input_labels, rel_types, masks = labels
        batch_size, sequence_length, vocab_size = decoder_output.shape
        vocab_masks_repeated = vocab_masks.reshape(
            [1, 1, -1]).repeat([batch_size, sequence_length, 1])
        pred = (decoder_output+vocab_masks_repeated).argmax(-1)

        print("\n预测结果:")
        preds_tokens = []
        for i in range(batch_size):
            pred_mask_token_ids = pred[i][mask_start_end[i][0]:mask_start_end[i][1]+1]
            pred_token = "".join(self.tokenizer.decode(pred_mask_token_ids).split(" "))
            preds_tokens.append(pred_token)
        print(preds_tokens)
        pred = []
        for i in range(batch_size):
            pred.append(self._get_event_type(preds_tokens[i]))

        label = rel_types.cpu()
        pred = torch.Tensor(pred)

        label_index = torch.where(label != 0)
        pred_index = torch.where(pred != 0)

        precision_micro = 0.0
        recall_micro = 0.0
        f1_micro = 0.0
        precision_macro = 0.0
        recall_macro = 0.0
        f1_macro = 0.0

        if len(label_index[0]) != 0 or len(pred_index[0]) != 0 :
            if len(pred_index[0]) != 0:
                label_precision = label[pred_index]
                pred_precision = pred[pred_index]
                precision_micro = precision_score(label_precision, pred_precision, average="micro", zero_division=0)
                precision_macro = precision_score(label_precision, pred_precision, average="macro", zero_division=0)

                if "label_precision" in metrics:
                    metrics["label_precision"].extend(label_precision.tolist())
                else:
                    metrics["label_precision"] = label_precision.tolist()
                if "pred_precision" in metrics:
                    metrics["pred_precision"].extend(pred_precision.tolist())
                else:
                    metrics["pred_precision"] = pred_precision.tolist()

            if len(label_index[0]) != 0:
                label_recall = label[label_index]
                pred_recall = pred[label_index]
                recall_micro = recall_score(label_recall, pred_recall, average="micro", zero_division=0)
                recall_macro = recall_score(label_recall, pred_recall, average="macro", zero_division=0)

                if "label_recall" in metrics:
                    metrics["label_recall"].extend(label_recall.tolist())
                else:
                    metrics["label_recall"] = label_recall.tolist()
                if "pred_recall" in metrics:
                    metrics["pred_recall"].extend(pred_recall.tolist())
                else:
                    metrics["pred_recall"] = pred_recall.tolist()

        f1_micro = 2 * precision_micro * recall_micro / ( precision_micro + recall_micro + 1e-9)
        f1_macro = 2 * precision_macro * recall_macro / ( precision_macro + recall_macro + 1e-9)

        batch_metrics = {
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            }   
        return metrics, batch_metrics

new_prompt_metrics_cal_func = metrics_cal_func

class BatchMetricsFunc:
    def __init__(self,rel2ids) -> None:
        self.rel2ids = rel2ids

    def __call__(self, labels: torch.Tensor, preds: Tensor, metrics: Dict[str, List[int]], trainer):
        label = labels.reshape([-1]).cpu()
        pred = preds.argmax(1).reshape([-1]).cpu()
        label_index = torch.where(label != 0)
        pred_index = torch.where(pred != 0)

        precision_micro = 0.0
        recall_micro = 0.0
        f1_micro = 0.0
        precision_macro = 0.0
        recall_macro = 0.0
        f1_macro = 0.0

        if len(label_index[0]) != 0 or len(pred_index[0]) != 0 :
            if len(pred_index[0]) != 0:
                label_precision = label[pred_index]
                pred_precision = pred[pred_index]
                precision_micro = precision_score(label_precision, pred_precision, average="micro", zero_division=0)
                precision_macro = precision_score(label_precision, pred_precision, average="macro", zero_division=0)

                if "label_precision" in metrics:
                    metrics["label_precision"].extend(label_precision.tolist())
                else:
                    metrics["label_precision"] = label_precision.tolist()
                if "pred_precision" in metrics:
                    metrics["pred_precision"].extend(pred_precision.tolist())
                else:
                    metrics["pred_precision"] = pred_precision.tolist()

            if len(label_index[0]) != 0:
                label_recall = label[label_index]
                pred_recall = pred[label_index]
                recall_micro = recall_score(label_recall, pred_recall, average="micro", zero_division=0)
                recall_macro = recall_score(label_recall, pred_recall, average="macro", zero_division=0)

                if "label_recall" in metrics:
                    metrics["label_recall"].extend(label_recall.tolist())
                else:
                    metrics["label_recall"] = label_recall.tolist()
                if "pred_recall" in metrics:
                    metrics["pred_recall"].extend(pred_recall.tolist())
                else:
                    metrics["pred_recall"] = pred_recall.tolist()

        f1_micro = 2 * precision_micro * recall_micro / ( precision_micro + recall_micro + 1e-9)
        f1_macro = 2 * precision_macro * recall_macro / ( precision_macro + recall_macro + 1e-9)

        batch_metrics = {
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            }
        return metrics, batch_metrics

def get_optimizer(model, lr: float):
    if isinstance(model,LSDRModel):
        optimizer = torch.optim.AdamW([
            {"params": model.mlm.parameters(), "lr": lr/5},
            {"params": model.fc.parameters()}
        ], lr=lr)
        return optimizer
    if isinstance(model, (NewPromptModel)):
        optimizer = torch.optim.AdamW([
            # {"params": model.mlm.parameters(), "lr": lr},
            # {"params": model.fc.parameters()},
            # {"params": model.attention.parameters()},
            # {"params": model.classification.parameters()},
            # {"params":model.layer_norm.parameters()}
            {"params": model.parameters()}
        ], lr=lr)
        return optimizer
