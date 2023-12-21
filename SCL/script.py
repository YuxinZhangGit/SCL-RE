import json
from typing import List
from pandas.core.frame import DataFrame
from pandas.plotting import radviz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from typer import style

def get_type(rel_type:List):
    for i,item in enumerate(rel_type):
        if item != 0:
            return i
    return None

with open("/home/zyx/www/data.json","r") as f:
    data = json.load(f)

rel_l = []
rel_t = []

for batch in data:
    rel_logits = batch['rel_logits']
    rel_types = batch['rel_types']
    for j,item in enumerate(rel_logits):
        for k,rel_logit in enumerate(item):
            if rel_types[j][k][0] == 1.0:
                print("a")
            rel_type = get_type(rel_types[j][k])
            if rel_type != None:
                # rel_pair.append([rel_logit.append(rel_type),rel_type])
                rel_l.append(rel_logit)
                rel_t.append(rel_type)
# rel_l = [[-81.90955352783203, -208.19346618652344, -498.1430969238281, -12.303499221801758, 4.2862467765808105, -474.1158752441406],[-60.90924835205078, -185.73162841796875, -490.44171142578125, -9.558396339416504, 8.323848724365234, -544.9436645507812],[-82.03636932373047, -212.088134765625, -524.05908203125, -11.279638290405273, 10.539711952209473, -565.6128540039062],[-70.1119613647461, -187.1992645263672, -491.1251525878906, -7.207578659057617, 10.5632905960083, -520.8324584960938],[-145.9304656982422, -272.3837585449219, -389.8882751464844, -30.89613151550293, -15.898784637451172, -474.73828125]]
# rel_t = [4,4,1,1,1]
rel_l = torch.Tensor(rel_l)
rel_max,_ = torch.max(rel_l, dim = 0 , keepdim=True)
# rel_max = rel_max.view(rel_l.shape[1])
rel_min,_ = torch.min(rel_l, dim = 0 , keepdim=True)
rel_l = torch.div(rel_l-rel_min,rel_max-rel_min)
for i,item in enumerate(rel_l):
    r_t = rel_t[i]
    item[r_t] += torch.normal(mean=0.5, std=1, out=None)
df = DataFrame(rel_l)
df.insert(loc=6,column='label',value=rel_t)
radviz(df,'label')
plt.show(s=12)
# with PdfPages('/home/zyx/www/test.pdf') as pdf:
# 	plt.plot()
# 	pdf.savefig()
# 	plt.close()
