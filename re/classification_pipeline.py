import json
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Dropout, Linear, CrossEntropyLoss
from transformers.models.bert import BertModel, BertConfig
from transformers.models.roberta import RobertaModel, RobertaConfig
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import *
from classification_model import ClassificationBaselineModel, NewPromptModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.file_utils import PaddingStrategy
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
import Levenshtein

#multi版本做prompt关系单分类

class NewPromptPipeline(object):
    def __init__(self, event2id: Dict[str, int], model_path: str, device: int, mlm_name_or_path: str, batch_size: int) -> None:
        self.model: NewPromptModel = NewPromptModel(
            mlm_name_or_path, event2id)
        with open(model_path, "rb") as f:
            self.model.load_state_dict(torch.load(f, map_location="cpu"))
        self.model.eval()

        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            mlm_name_or_path)
        self.event2id = event2id
        self.device = torch.device(device)
        self.model = self.model.to(device=self.device)
        self.batch_size = batch_size
        self.id2event: Dict[int, str] = {}
        for k, v in event2id.items():
            self.id2event[v] = k
        self.key_ids: Set[int] = set()
        self.event_type_num = len(self.event2id)
        self.event_types = []
        for i in range(self.event_type_num):
            self.event_types.append(self.id2event[i])
        # self.tokenizer=BertTokenizer.from_pretrained(mlm_name_or_path)
        
        self.max_event_size=0
        self.candidate_char_set:Set[str]=set()
        for event in event2id:
            self.max_event_size=max(self.max_event_size,len(event))
            for char in event:
                self.candidate_char_set.add(char)
        self.candidate_char_set.add('。')
        
        self.vocab_mask_size=len(self.candidate_char_set)
        self.vocab_mask=torch.zeros([18000 if "ernie" in mlm_name_or_path else self.tokenizer.vocab_size],dtype=torch.float32)
        # self.vocab_mask=torch.zeros([ self.tokenizer.vocab_size],dtype=torch.float32)
        self.vocab_mask.fill_(float("-inf"))
        for char in self.candidate_char_set:
            ind=self.tokenizer.encode(char,add_special_tokens=False)[0]
            self.vocab_mask[ind]=1.0
        self.vocab_mask[self.tokenizer.pad_token_id]=1.0
    def _get_event_type(self,s:str):
        t=torch.zeros([self.event_type_num],dtype=torch.float)
        for i in range(self.event_type_num):
            t[i]=Levenshtein.jaro(s,self.event_types[i])
        return t.argmax().item()
    def make_prompt(self,text:str,company:str):
        res=text
        
        if not res.endswith(('。',';','!','?')):
            res+='。'
        res+=(f"{company}发生了：")
        
        input_ids=res+"".join([self.tokenizer.mask_token for i in range(self.max_event_size+1)])
        
        return input_ids
    def _predict_batch(self, textIds: List[str], texts: List[str],companies):
        batch_size=len(textIds)
        input_tokens=[]

        for textId,text,company in zip(textIds,texts,companies):
            # input_id=self.make_prompt(text,company)
            input_tokens.append(text)
        tokenizer_output=self.tokenizer(input_tokens,padding=PaddingStrategy.LONGEST,truncation=True,max_length=512,return_attention_mask=True)
        input_ids:List[List[int]]=tokenizer_output["input_ids"]
        masks=tokenizer_output["attention_mask"]
        res = []
        mask_start_end=[]
        masks4masks=[]
        for i in range(batch_size):
            mask_start=input_ids[i].index(self.tokenizer.mask_token_id)
            mask_end=mask_start+self.max_event_size-1
            mask_start_end.append([mask_start,mask_end])
            mask4mask=[0 for j in range(mask_start+1)]+[1 for j in range(self.max_event_size)]
            mask4mask.extend([0 for j in range(len(input_ids[0])-len(mask4mask))])
            masks4masks.append(mask4mask)
        masks4masks=torch.tensor(mask4mask,dtype=torch.long,device=self.device)
        mask_start_end=torch.tensor(mask_start_end,dtype=torch.long,device=self.device)
        input_ids=torch.tensor(input_ids,dtype=torch.long,device=self.device)
        masks=torch.tensor(masks,dtype=torch.long,device=self.device)
        decoder_output=self.model(input_ids,masks)
        batch_size,sequence_length,vocab_size=decoder_output.shape
        vocab_masks_repeated=self.vocab_mask.to(self.device).reshape([1, 1, -1]).repeat([batch_size, sequence_length, 1])
        pred=(decoder_output+vocab_masks_repeated).argmax(-1)
        preds_tokens=[]
        for i in range(batch_size):
            pred_mask_token_ids=pred[i][mask_start_end[i][0]:mask_start_end[i][1]+1]
            pred_token="".join(self.tokenizer.decode(pred_mask_token_ids).split(" ")).replace(self.tokenizer.pad_token,"")
            preds_tokens.append(pred_token)
        preds=[]
        for i in range(batch_size):

            preds.append(self._get_event_type(preds_tokens[i]))
            # print("".join(self.tokenizer.decode(pred[i]).split(" ")))
            # print(pred_token)
        for i in preds:
            res.append(self.id2event[i])
        return res
    def __call__(self, data: List[Dict[str, Union[str, List[str]]]]) -> Any:
        data1 = []
        for item in data:
            
            textId = item["textId"]
            companies = item["companies"]
            text = item["text"]
            
            for company in companies:
                if len(text)>450:
                    if company in text[:400]:
                        prompt=self.make_prompt(text[:400],company)
                        data1.append({
                            "textId": textId,
                            "text": prompt,
                            "company":company
                    })
                    else:
                        prompt=self.make_prompt(text[350:],company)
                        data1.append({
                            "textId": textId,
                            "text": prompt,
                            "company":company
                    })
                else:
                    prompt=self.make_prompt(text,company)
                    data1.append({
                        "textId": textId,
                        "text": prompt,
                        "company":company
                    })
                    

        res = []
        res_dict = {}
        dataloader: DataLoader = DataLoader(
            dataset=data1,
            batch_size=self.batch_size,
            shuffle=False, num_workers=12)
        it = iter(dataloader)
        with tqdm(total=len(dataloader), ncols=80) as tqbar:
            with torch.no_grad():
                while True:
                    batch_data = None
                    try:
                        batch_data = next(it)
                    except Exception:
                        break
                    batch_textIds = batch_data["textId"]
                    batch_texts = batch_data["text"]
                    batch_companies=batch_data["company"]
                    batch_size = len(batch_textIds)
                    batch_result: List[str] = self._predict_batch(
                        batch_textIds, batch_texts,batch_companies)
                    for i in range(batch_size):
                        textId = batch_textIds[i]
                        eventType = batch_result[i]
                        company=batch_companies[i]
                        if textId not in res_dict:
                            res_dict[textId] = {
                                "textId": textId,
                                "eventTags": []
                            }
                        res_dict[textId]["eventTags"].append({
                            "eventType": eventType,
                            "eventCompany": company
                        })
                    tqbar.update(1)
        for item in data:
            if item["textId"] in res_dict:
                res.append(res_dict[item["textId"]])
        return res
if __name__ == "__main__":
    
    event2prompt = {}
    with open("/data/zyx/ccks2/event2prompt.json") as f:
        event2prompt=json.load(f)
    event2id = {}
    with open("/data/zyx/ccks2/key_events.json","r") as f:
        key_events=json.load(f)
        for i,eve in enumerate(key_events):
            key_events[i] = event2prompt[eve]
    with open("/data/zyx/ccks2/event2id.json", "r") as f:
        event2id = json.load(f)
        ids=[]
        events=[]
        for k,v in event2id.items():
            ids.append(v)
            events.append(event2prompt[k])
        event2id=dict(zip(events,ids))

    key_counts=[]
    pipeline=NewPromptPipeline(event2id,"/data/zyx/ccks2/new_prompt_output_final2/epoch7.pt",3,"nghuyong/ernie-1.0",24)
    data = []
    with open("/data/zyx/ccks2/data/test_nered.json", "r") as f:
        data = json.load(f)
    res = pipeline(data)
    key_count=0
    for obj in res:
        for event in obj["eventTags"]:
            if event["eventType"] in key_events:
                key_count+=1
            for k,v in event2prompt.items():
                if event["eventType"]==v:
                    event["eventType"]=k
                    break
    print("key_event_num:"+str(key_count))
    key_counts.append(key_count)
    with open(f"/data/zyx/ccks2/data/zyx_valid_result1.csv", "w") as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")