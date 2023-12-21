from html import entities
from typing import *
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler,Sampler
from transformers import AutoModel,AutoTokenizer, BertTokenizerFast, RobertaTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.tokenization_utils_base import TruncationStrategy


@dataclass
class ClassificationInputfeature:
    # textId:str
    text:str
    head_ent:str
    tail_ent:str
    relations:int

class ClassificationCollator(object):
    def __init__(self,tokenizer:Union[BertTokenizer,RobertaTokenizer],event2id:Dict[str,int]) -> None:
        self.tokenizer=tokenizer
        self.event2id=event2id
    def _naive_sequence_search(self,seq:List[int],pattern_seq:List[int]):
        assert len(seq)>=len(pattern_seq)
        res=-1
        seq_size=len(seq)
        pattern_size=len(pattern_seq)
        for i in range(seq_size):
            flag=True
            for j in range(pattern_size):
                if i+j>=seq_size:
                    break
                if seq[i+j]!=pattern_seq[j]:
                    flag=False
                    break
            if flag:
                return i
        return res
    def __call__(self, batch_data:List[ClassificationInputfeature]) -> Any:
        batch_size=len(batch_data)
        texts=[]
        labels=[]
        for inputfeature in batch_data:
            texts.append(inputfeature.text)
            labels.append(self.event2id[inputfeature.event_type])
        tokenizer_output=self.tokenizer(
            texts,
            padding=PaddingStrategy.LONGEST,
            max_length=512,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids:torch.Tensor=tokenizer_output["input_ids"]
        masks:torch.Tensor=tokenizer_output["attention_mask"]
        token_type_ids=torch.zeros(input_ids.size(),dtype=torch.long)
        labels=torch.tensor(labels,dtype=torch.long)
        for i in range(batch_size):
            company_id=self.tokenizer.encode( batch_data[i].company,add_special_tokens=False)
            index=self._naive_sequence_search(input_ids[i].tolist(),company_id)
            if index==-1:
                print(f"\n公司 {batch_data[i].company} 不在文本里")
                # print(f"\nid为 {batch_data[i].textId} 的样本,公司 {batch_data[i].company} 不在文本里")
                # token_type_ids[i,:masks[i].sum().item()]=1
            else :
                token_type_ids[i,index:index+len(company_id)]=1
        return input_ids,masks,token_type_ids,labels

class LSDRCollator(object):
    def __init__(self,tokenizer:Union[BertTokenizer,RobertaTokenizer]) -> None:
        self.tokenizer=tokenizer

    def __call__(self, batch_data:List[ClassificationInputfeature]) -> Any:
        batch_size=len(batch_data)
        texts=[]
        head_ents=[]
        tail_ents=[]
        labels=[]
        for inputfeature in batch_data:
            texts.append(inputfeature.text)
            labels.append(inputfeature.relations)
            head_ents.append(inputfeature.head_ent)
            tail_ents.append(inputfeature.tail_ent)
        # 文本转input_ids
        tokenizer_output=self.tokenizer(
            texts,
            padding=PaddingStrategy.LONGEST,
            max_length=512,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids:torch.Tensor=tokenizer_output["input_ids"]
        masks:torch.Tensor=tokenizer_output["attention_mask"]
        # 头实体转input_ids
        tokenizer_output=self.tokenizer(
            head_ents,
            padding=PaddingStrategy.LONGEST,
            max_length=64,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ent_h_ids=tokenizer_output["input_ids"]
        ent_h_masks=tokenizer_output["attention_mask"]
        # 尾实体转input_ids
        tokenizer_output=self.tokenizer(
            tail_ents,
            padding=PaddingStrategy.LONGEST,
            max_length=64,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ent_t_ids=tokenizer_output["input_ids"]
        ent_t_masks=tokenizer_output["attention_mask"]

        labels=torch.tensor(labels,dtype=torch.long)

        return input_ids,masks,ent_h_ids,ent_h_masks,ent_t_ids,ent_t_masks,labels

class NewPromptCollator(object):
    def __init__(self,mlm_name_or_path,rel2ids_zh:Dict[str,int]) -> None:
        self.tokenizer=BertTokenizer.from_pretrained(mlm_name_or_path)
        self.rel2ids_zh=rel2ids_zh
        self.max_rel_size=2
        self.candidate_char_set:Set[str]=set()
        self.id2rel_zh=[]
        for rel in rel2ids_zh:
            self.id2rel_zh.append(rel)
            for char in rel:
                self.candidate_char_set.add(char)
        # self.candidate_char_set.add('。')
        
        self.vocab_mask_size=len(self.candidate_char_set)
        self.vocab_mask=torch.zeros(self.tokenizer.vocab_size,dtype=torch.float32)
        self.vocab_mask.fill_(float("-inf"))
        for char in self.candidate_char_set:
            ind=self.tokenizer.encode(char,add_special_tokens=False)[0]
            self.vocab_mask[ind]=1.0
        # self.vocab_mask[self.tokenizer.pad_token_id]=1.0
    def make_prompt(self,text:str,ent_h:str,ent_t:str,rel:int):
        res=text
        if not res.endswith(('。',';','!','?')):
            res+='。'
        res+=(f"{ent_h}")        
        input_ids=res+"".join([self.tokenizer.mask_token for i in range(self.max_rel_size)])+(f"{ent_t}")+'。'
        input_labels=res+self.id2rel_zh[rel]+(f"{ent_t}")+'。'
        return input_ids,input_labels
    def __call__(self,batch_data:List[ClassificationInputfeature]) -> Any:
        batch_size=len(batch_data)
        input_tokens=[]
        input_labels=[]
        rel_types=[]

        for i,inputfeature in enumerate(batch_data):
            text=inputfeature.text
            input_id,input_label=self.make_prompt(text,inputfeature.head_ent,inputfeature.tail_ent,inputfeature.relations)
            input_tokens.append(input_id)
            input_labels.append(input_label)
            rel_types.append(inputfeature.relations)
        tokenizer_output=self.tokenizer(input_tokens,padding=PaddingStrategy.LONGEST,truncation=True,max_length=512,return_attention_mask=True)
        input_ids:List[List[int]]=tokenizer_output["input_ids"]
        masks=tokenizer_output["attention_mask"]
        tokenizer_output=self.tokenizer(input_labels,padding=PaddingStrategy.MAX_LENGTH,max_length=len(input_ids[0]))
        input_labels=tokenizer_output["input_ids"]
        mask_start_end=[]
        masks4masks=[]
        
        for i in range(batch_size):
            if len(input_labels[i])!=len(input_ids[i]):
                print("异常！",batch_data[i])
            assert len(input_labels[i])==len(input_ids[i])
            mask_start=input_ids[i].index(self.tokenizer.mask_token_id)
            mask_end=mask_start+1
            mask_start_end.append([mask_start,mask_end])
            mask4mask=[0 for j in range(mask_start)]+[1 for j in range(self.max_rel_size)]
            mask4mask.extend([0 for j in range(len(input_ids[0])-len(mask4mask))])
            masks4masks.append(mask4mask)
            
        masks4masks=torch.tensor(mask4mask,dtype=torch.long)
        mask_start_end=torch.tensor(mask_start_end,dtype=torch.long)
        input_labels=torch.tensor(input_labels,dtype=torch.long)
        input_ids=torch.tensor(input_ids,dtype=torch.long)
        masks=torch.tensor(masks,dtype=torch.long)
        rel_types=torch.tensor(rel_types,dtype=torch.long)
        
        
        return input_ids,masks, masks4masks,mask_start_end,self.vocab_mask,input_labels,rel_types
# class ClassificationSampler(Sampler):
#     def __init__(self, data_source: List[ClassificationInputfeature],event2id:Dict[str,int],key_events:List[str],replacement:bool=True) -> None:
#         super().__init__(data_source)   
#         self.event2id=event2id
#         self.key_events=set(key_events)
#         self.key_id=[self.event2id[event] for event in self.key_events]
#         self.data=data_source
#         self.replacement=replacement
#         # self.type_counts={}
#         self.weights=torch.ones([len(self.data)],dtype=torch.float32)
#         # for inputfeature in self.data:
#         #     if inputfeature.event_type in self.type_counts:
#         #         self.type_counts[inputfeature.event_type]+=1
#         #     else:
#         #         self.type_counts[inputfeature.event_type]=1
#         for i,inputfeature in enumerate(self.data):
#             if inputfeature.event_type in self.key_events:
#                 self.weights[i]=0.5 # 1.0 # 6.0
#         self._impl=WeightedRandomSampler(weights=self.weights,num_samples=int(len(data_source)/5),replacement=replacement)
    def __iter__(self):
        return self._impl.__iter__()
    
    def __len__(self)->int:
        return len(self._impl)