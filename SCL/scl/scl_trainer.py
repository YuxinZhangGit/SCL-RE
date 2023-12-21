import argparse
from ast import arg
import math
import os
from typing import Type

import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, BertConfig
from transformers import BertTokenizer

from www import models, prediction
from www import sampling
from www import util
from www.entities import Dataset
from www.evaluator import Evaluator
from www.input_reader import JsonInputReader, BaseInputReader
from www.loss import WWWLoss, Loss
from tqdm import tqdm
from www.trainer import BaseTrainer
import json

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
'''
    模型最主要的训练器对象
    :param BaseTrainer: 训练器基本方法对象（读写存储log，保存模型等设置）具体可以参考trainer.py文件
    :return: 返回一个训练器对象（包含所有的训练所需参数）
'''
class WWWTrainer(BaseTrainer):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # token 编码器（Transformers接口）
        self._tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path,
                                                        do_lower_case=args.lowercase,
                                                        cache_dir=args.cache_path)
    # 训练流程
    def train(self, train_path: str, valid_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        train_label, valid_label = 'train', 'valid'

        self._logger.info("Datasets: %s, %s" % (train_path, valid_path))
        self._logger.info("Model type: %s" % args.model_type)

        # 创建log CSV文件
        self._init_train_logging(train_label)
        self._init_eval_logging(valid_label)
        self.data = []

        # 读取数据
        input_reader = input_reader_cls(types_path, self._tokenizer, args.neg_entity_count,
                                        args.neg_relation_count, args.max_span_size, self._logger)
        train_dataset = input_reader.read(train_path, train_label)
        validation_dataset = input_reader.read(valid_path, valid_label)
        self._log_datasets(input_reader)

        train_sample_count = train_dataset.document_count
        updates_epoch = train_sample_count // args.train_batch_size
        updates_total = updates_epoch * args.epochs

        self._logger.info("Updates per epoch: %s" % updates_epoch)
        self._logger.info("Updates total: %s" % updates_total)

        # 加载模型（如果是在训练好的模型基础上）
        model = self._load_model(input_reader)
        # 模型移到同一个设备上
        model.to(self._device)

        # 创建优化器对象
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, correct_bias=False)
        # 创建优化策略对象
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=args.lr_warmup * updates_total,
                                                                 num_training_steps=updates_total)
        # 创建计算损失对象（这里的代码在损失计算后直接backward，所以需要同时传入优化器和计划对象）
        rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        compute_loss = WWWLoss(rel_criterion, entity_criterion, model, optimizer, scheduler, args.max_grad_norm)

        if args.init_eval:
            self._eval(model, validation_dataset, input_reader, 0, updates_epoch)

        # 训练流程
        for epoch in range(args.epochs):
            # 进入epoch训练流程
            self._train_epoch(model, compute_loss, optimizer, train_dataset, updates_epoch, epoch)

            #============================================= eval validation sets ================================================#
            if not args.final_eval or (epoch == args.epochs - 1) or (epoch != 0 and epoch % 1 == 0):
                self._eval(model, validation_dataset, input_reader, epoch + 1, updates_epoch)

        # 保存最终的模型
        with open("/home/zyx/www/data_19.json","w") as f:
            json.dump(self.data,f)
        extra = dict(epoch=args.epochs, updates_epoch=updates_epoch, epoch_iteration=0)
        global_iteration = args.epochs * updates_epoch
        self._save_model(self._save_path, model, self._tokenizer, global_iteration,
                         optimizer=optimizer if self._args.save_optimizer else None, extra=extra,
                         include_iteration=False, name='final_model')

        self._logger.info("Logged in: %s" % self._log_path)
        self._logger.info("Saved in: %s" % self._save_path)
        self._close_summary_writer()

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args
        dataset_label = 'test'

        self._logger.info("Dataset: %s" % dataset_path)
        self._logger.info("Model: %s" % args.model_type)

        # create log csv files
        self._init_eval_logging(dataset_label)

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size, logger=self._logger)
        test_dataset = input_reader.read(dataset_path, dataset_label)
        self._log_datasets(input_reader)

        # load model
        model = self._load_model(input_reader)
        model.to(self._device)

        # evaluate
        self._eval(model, test_dataset, input_reader)

        self._logger.info("Logged in: %s" % self._log_path)
        self._close_summary_writer()

    def predict(self, dataset_path: str, types_path: str, input_reader_cls: Type[BaseInputReader]):
        args = self._args

        # read datasets
        input_reader = input_reader_cls(types_path, self._tokenizer,
                                        max_span_size=args.max_span_size,
                                        spacy_model=args.spacy_model)
        dataset = input_reader.read(dataset_path, 'dataset')

        model = self._load_model(input_reader)
        model.to(self._device)

        self._predict(model, dataset, input_reader)
    # 加载已有模型
    def _load_model(self, input_reader):
        model_class = models.get_model(self._args.model_type)

        config = BertConfig.from_pretrained(self._args.model_path, cache_dir=self._args.cache_path)
        util.check_version(config, model_class, self._args.model_path)

        config.www = model_class.VERSION
        model = model_class.from_pretrained(self._args.model_path,
                                            config=config,
                                            # SpERT model parameters
                                            cls_token=self._tokenizer.convert_tokens_to_ids('[CLS]'),
                                            relation_types=input_reader.relation_type_count - 1,
                                            entity_types=input_reader.entity_type_count,
                                            max_pairs=self._args.max_pairs,
                                            prop_drop=self._args.prop_drop,
                                            size_embedding=self._args.size_embedding,
                                            freeze_transformer=self._args.freeze_transformer,
                                            cache_dir=self._args.cache_path)

        return model
    # epoch训练过程
    def _train_epoch(self, model: torch.nn.Module, compute_loss: Loss, optimizer: Optimizer, dataset: Dataset,
                     updates_epoch: int, epoch: int):
        self._logger.info("Train epoch: %s" % epoch)
        args = self._args
        # 创建数据加载器
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.train_batch_size, shuffle=True, drop_last=True,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        model.zero_grad()

        iteration = 0
        total = dataset.document_count // self._args.train_batch_size
        for batch in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            batch = util.to_device(batch, self._device)

            # forward 步骤
            entity_logits, rel_logits, rel_repr = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                                              entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                                              relations=batch['rels'], rel_masks=batch['rel_masks'])

            # 计算损失以及优化参数
            batch_loss = compute_loss.compute(entity_logits=entity_logits, rel_logits=rel_logits,rel_repr=rel_repr,
                                              rel_types=batch['rel_types'], entity_types=batch['entity_types'],
                                              entity_sample_masks=batch['entity_sample_masks'],
                                              rel_sample_masks=batch['rel_sample_masks'])

            # 写日志
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration
            if epoch == args.epochs - 1:
                self.data.append({"rel_logits":rel_repr.cpu().tolist(),"rel_types":batch['rel_types'].cpu().tolist()})

            if global_iteration % self._args.train_log_iter == 0:
                self._log_train(optimizer, batch_loss, epoch, iteration, global_iteration, dataset.label)

        return iteration

    def _eval(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader,
              epoch: int = 0, updates_epoch: int = 0, iteration: int = 0):
        self._logger.info("Evaluate: %s" % dataset.label)

        if isinstance(model, DataParallel):
            model = model.module

        # 创建一个评估器
        predictions_path = os.path.join(self._log_path, f'predictions_{dataset.label}_epoch_{epoch}.json')
        examples_path = os.path.join(self._log_path, f'examples_%s_{dataset.label}_epoch_{epoch}.html')
        evaluator = Evaluator(dataset, input_reader, self._tokenizer,
                              self._args.rel_filter_threshold, self._args.no_overlapping, predictions_path,
                              examples_path, self._args.example_count)

        # 创建数据加载器
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        with torch.no_grad():
            # 评估阶段使用的eval() 不更新参数
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # 运行模型
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result

                # batch验证
                evaluator.eval_batch(entity_clf, rel_clf, rels, batch)

        global_iteration = epoch * updates_epoch + iteration
        ner_eval, rel_eval, rel_nec_eval = evaluator.compute_scores()
        self._log_eval(*ner_eval, *rel_eval, *rel_nec_eval,
                       epoch, iteration, global_iteration, dataset.label)

        if self._args.store_predictions and not self._args.no_overlapping:
            evaluator.store_predictions()

        if self._args.store_examples:
            evaluator.store_examples()

    def _predict(self, model: torch.nn.Module, dataset: Dataset, input_reader: BaseInputReader):
        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(dataset, batch_size=self._args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=self._args.sampling_processes, collate_fn=sampling.collate_fn_padding)

        pred_entities = []
        pred_relations = []

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / self._args.eval_batch_size)
            for batch in tqdm(data_loader, total=total, desc='Predict'):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                result = model(encodings=batch['encodings'], context_masks=batch['context_masks'],
                               entity_masks=batch['entity_masks'], entity_sizes=batch['entity_sizes'],
                               entity_spans=batch['entity_spans'], entity_sample_masks=batch['entity_sample_masks'],
                               inference=True)
                entity_clf, rel_clf, rels = result

                # convert predictions
                predictions = prediction.convert_predictions(entity_clf, rel_clf, rels,
                                                             batch, self._args.rel_filter_threshold,
                                                             input_reader)

                batch_pred_entities, batch_pred_relations = predictions
                pred_entities.extend(batch_pred_entities)
                pred_relations.extend(batch_pred_relations)

        prediction.store_predictions(dataset.documents, pred_entities, pred_relations, self._args.predictions_path)

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _log_train(self, optimizer: Optimizer, loss: float, epoch: int,
                   iteration: int, global_iteration: int, label: str):
        # average loss
        avg_loss = loss / self._args.train_batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # log to tensorboard
        self._log_tensorboard(label, 'loss', loss, global_iteration)
        self._log_tensorboard(label, 'loss_avg', avg_loss, global_iteration)
        self._log_tensorboard(label, 'lr', lr, global_iteration)

        # log to csv
        self._log_csv(label, 'loss', loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'loss_avg', avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, 'lr', lr, epoch, iteration, global_iteration)

    def _log_eval(self, ner_prec_micro: float, ner_rec_micro: float, ner_f1_micro: float,
                  ner_prec_macro: float, ner_rec_macro: float, ner_f1_macro: float,

                  rel_prec_micro: float, rel_rec_micro: float, rel_f1_micro: float,
                  rel_prec_macro: float, rel_rec_macro: float, rel_f1_macro: float,

                  rel_nec_prec_micro: float, rel_nec_rec_micro: float, rel_nec_f1_micro: float,
                  rel_nec_prec_macro: float, rel_nec_rec_macro: float, rel_nec_f1_macro: float,
                  epoch: int, iteration: int, global_iteration: int, label: str):

        # log to tensorboard
        self._log_tensorboard(label, 'eval/ner_prec_micro', ner_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_micro', ner_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_micro', ner_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_prec_macro', ner_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_recall_macro', ner_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/ner_f1_macro', ner_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_prec_micro', rel_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_micro', rel_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_micro', rel_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_prec_macro', rel_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_recall_macro', rel_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_f1_macro', rel_f1_macro, global_iteration)

        self._log_tensorboard(label, 'eval/rel_nec_prec_micro', rel_nec_prec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_micro', rel_nec_rec_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_micro', rel_nec_f1_micro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_prec_macro', rel_nec_prec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_recall_macro', rel_nec_rec_macro, global_iteration)
        self._log_tensorboard(label, 'eval/rel_nec_f1_macro', rel_nec_f1_macro, global_iteration)

        # log to csv
        self._log_csv(label, 'eval', ner_prec_micro, ner_rec_micro, ner_f1_micro,
                      ner_prec_macro, ner_rec_macro, ner_f1_macro,

                      rel_prec_micro, rel_rec_micro, rel_f1_micro,
                      rel_prec_macro, rel_rec_macro, rel_f1_macro,

                      rel_nec_prec_micro, rel_nec_rec_micro, rel_nec_f1_micro,
                      rel_nec_prec_macro, rel_nec_rec_macro, rel_nec_f1_macro,
                      epoch, iteration, global_iteration)

    def _log_datasets(self, input_reader):
        self._logger.info("Relation type count: %s" % input_reader.relation_type_count)
        self._logger.info("Entity type count: %s" % input_reader.entity_type_count)

        self._logger.info("Entities:")
        for e in input_reader.entity_types.values():
            self._logger.info(e.verbose_name + '=' + str(e.index))

        self._logger.info("Relations:")
        for r in input_reader.relation_types.values():
            self._logger.info(r.verbose_name + '=' + str(r.index))

        for k, d in input_reader.datasets.items():
            self._logger.info('Dataset: %s' % k)
            self._logger.info("Document count: %s" % d.document_count)
            self._logger.info("Relation count: %s" % d.relation_count)
            self._logger.info("Entity count: %s" % d.entity_count)

    def _init_train_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'lr': ['lr', 'epoch', 'iteration', 'global_iteration'],
                                        'loss': ['loss', 'epoch', 'iteration', 'global_iteration'],
                                        'loss_avg': ['loss_avg', 'epoch', 'iteration', 'global_iteration']})

    def _init_eval_logging(self, label):
        self._add_dataset_logging(label,
                                  data={'eval': ['ner_prec_micro', 'ner_rec_micro', 'ner_f1_micro',
                                                 'ner_prec_macro', 'ner_rec_macro', 'ner_f1_macro',
                                                 'rel_prec_micro', 'rel_rec_micro', 'rel_f1_micro',
                                                 'rel_prec_macro', 'rel_rec_macro', 'rel_f1_macro',
                                                 'rel_nec_prec_micro', 'rel_nec_rec_micro', 'rel_nec_f1_micro',
                                                 'rel_nec_prec_macro', 'rel_nec_rec_macro', 'rel_nec_f1_macro',
                                                 'epoch', 'iteration', 'global_iteration']})
