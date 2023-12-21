## README

##### 文件说明
- requirements.txt 保存模型导入的第三方依赖包版本
- trainer.py 模型训练的主要文档，里面主要是训练+预测流程
- run_classification.py 普通RE模型运行入口+参数设置
- run_prompt.py 基于prompt的RE模型运行入口+参数设置
- classification_model.py 模型主体
- classification_data_process.py 模型数据处理
- model_save/ 保存模型训练好的文件

#### 已经配置好的环境
- 服务器
  - 47.110.73.22
- 将模型放到服务器
  - 上传到任意服务器路径下
  - 或可直接使用模型代码：
- 可直接使用的虚拟环境
```conda activate www_re```


#### 模型使用
建议使用服务器运行

##### 运行前检查

- 确认```run_classification.py```  ```run_prompt.py```文件中使用的多GPU编号
  - 执行前先使用```nvidia-smi```命令找到当前可用的GPU
  - 根据可用GPU编号```os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'```,其中```'0,1,2,3'```为运行模型时可以用的GPU编号
- 确认```run_classification.py```  ```run_prompt.py```目录下对应文件参数符合需求，其中需要重点关注的配置参数如下
  - train_path 训练文件，做了数据增强后可以修改该路径切换到对应的增强后的文件进行训练
  - valid_path 评估（测试）文件
  - types_path 定义的关系文件
  - train_batch_size 根据GPU数量进行修改，否则可能会爆显存，（当有4块GPU时，可设置为8；有两块GPU时，可设置为4）
  - eval_batch_size 
  - epochs （50、80、100）

##### 运行模型（以当前服务器上的模型为例）
- 进入到配置好的虚拟环境
```shell
conda activate www_re
```
- 当需要后台挂起指定输出流时，执行以下命令
```shell
nohup python -u run_classification.py  > myout.log  &  
nohup python -u run_prompt.py  > myout.log  &  
```
- 查看输出
  - ```myout.log``` 检查程序当前运行状态、运行阶段等
