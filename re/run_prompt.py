import functools
from html import entities
from lib2to3.pgen2.tokenize import tokenize
from typing import *

from transformers import AutoTokenizer, BertTokenizer
from trainer import Trainer
import torch
import json
import random
import argparse
import numpy as np
from classification_model import(
    NewPromptModel,
    NewPromptBatchMetricsFunc,
    new_prompt_batch_cal_loss_func,
    new_prompt_batch_forward_func,
    metrics_cal_func,
    get_optimizer
)
from classification_data_process import(
    ClassificationInputfeature,
    NewPromptCollator
    )
from trainer import Trainer
from torch.utils.data import RandomSampler, SequentialSampler
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def get_dataset(data:List[ClassificationInputfeature],neg_sample:float):
    rels4types = {}
    dataset=[]
    for input_f in data:
        if input_f.relations in rels4types:
            rels4types[input_f.relations].append(input_f)
        else:
            rels4types[input_f.relations] = [input_f]
    for type_,rels in rels4types.items():
        if type_ != 0:
            size = len(rels)
            random.shuffle(rels)
            dataset.extend(rels)

        else:
            size = len(rels)
            random.shuffle(rels)
            #可以设置负采样比例！
            rels = rels[:int( size * neg_sample )]
            size = len(rels)
            dataset.extend(rels)
    return dataset
def get_data(data_path):
    train_data: List[ClassificationInputfeature] = []
    with open(data_path, "r") as f:
        data = json.load(f)
        for text,item in data.items():
            rel_list = list(item["entities"].keys())
            for h_ent in rel_list:
                for t_ent in rel_list:
                    if h_ent == t_ent:
                        continue
                    train_data.append(ClassificationInputfeature(
                        text,
                        h_ent,
                        t_ent,
                        get_rel_type(h_ent,t_ent,item["relations"])
                    ))
    return train_data

import os
#=======================================================================================#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--main_device", type=int, default=0)
parser.add_argument("--device_ids", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-6)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--mlm_name_or_path", type=str,
                    default="bert-base-chinese")
parser.add_argument("--data_path_train", type=str, default="/home/zyx/dataset/data/train_only.json")
parser.add_argument("--data_path_valid", type=str, default="/home/zyx/www_re/data/re_mac_only.json")
parser.add_argument("--gradient_accumulate", type=int, default=1)


if __name__ == '__main__':
    set_random_seed(620526)
    args = parser.parse_args()
    for kwarg in args._get_kwargs():
        print(kwarg)
    """一些常规的设置"""
    dev = torch.device(args.main_device)
    device_ids = list(map(lambda x: int(x), args.device_ids.split(",")))
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.learning_rate
    epochs = args.epochs
    mlm_name_or_path = args.mlm_name_or_path
    data_path_train = args.data_path_train
    data_path_valid = args.data_path_valid
    gradient_accumulate = args.gradient_accumulate

    # 数据处理：这里的数据需要以一个三元组为例，也就是在前期需要进行头尾实体配对。
    # get relation ids - rel2id
    rel2ids = {"run":1,"use":2,"fun":3,"sat":4,"inc":5,"ope":6}
    rel2ids_zh = {"无关":0,"进行":1,"采用":2,"功能":3,"符合":4,"包含":5,"算子":6}

    def get_rel_type(h,t,rel_list):
        for rel in rel_list:
            if rel[0] == h and rel[2] == t:
                r = rel2ids[rel[1]]
                return r
        return 0
    train_data = get_data(data_path=data_path_train)
    valid_data = get_data(data_path=data_path_valid)
#=======================================================================================#
    training_dataset = get_dataset(train_data,0.5)
    valid_dataset = get_dataset(valid_data,1.0)
    print(str(0.5))

    tokenizer = AutoTokenizer.from_pretrained(mlm_name_or_path)
    collator=NewPromptCollator(mlm_name_or_path=mlm_name_or_path,rel2ids_zh=rel2ids_zh)

    model = NewPromptModel(mlm_name_or_path)
    # model=torch.nn.parallel.DataParallel(model,device_ids=device_ids)

    optimizer = get_optimizer(model, learning_rate)

    training_dataset_sampler = RandomSampler(training_dataset)
    valid_dataset_sampler = SequentialSampler(valid_dataset)

    batch_metrics_func=NewPromptBatchMetricsFunc(mlm_name_or_path,rel2ids_zh)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        output_dir="output",
        training_dataset=training_dataset,
        valid_dataset=valid_dataset,
        test_dataset=None,
        metrics_key="f1",
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        batch_forward_func=new_prompt_batch_forward_func,
        batch_cal_loss_func=new_prompt_batch_cal_loss_func,
        batch_metrics_func=batch_metrics_func,
        metrics_cal_func=metrics_cal_func,
        collate_fn=collator,
        device=dev,
        train_dataset_sampler=training_dataset_sampler,
        valid_dataset_sampler=valid_dataset_sampler,
        valid_step=1,
        start_epoch=0,
        gradient_accumulate=gradient_accumulate,
        save_model=True,
        save_model_steps= 1
    )
    trainer.train()
    for item in trainer.epoch_metrics:
        print (item)
