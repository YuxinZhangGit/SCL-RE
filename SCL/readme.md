## README

##### 文件说明
- configs/ 模型配置
- data/ 数据文件
  - datasets/ 数据集
    - train_all.json 模型训练数据集（包含非连续实体+实体对重叠）
    - test_all.json 模型测试数据集（包含非连续实体+实体对重叠）
    - train_only.json 模型训练数据集（简单数据集，不包含非连续实体+实体对重叠）
    - test_only.json 模型测试数据集（简单数据集，不包含非连续实体+实体对重叠）
  - log/ 模型log
  - models/ 最后用于测试和预测的模型文件
    - final_model 保存训练得到的模型
- requirements.txt 保存模型导入的第三方依赖包版本
- scl/ 模型结构文件
- args.py 参数
- config_reader.py 配置文件读取脚本
- scl.py 模型入口

#### 已经配置好的环境
- 服务器
  - xx.xx.xx.xx
- 将模型放到服务器
  - 上传到任意服务器路径下
  - 或可直接使用模型代码：
- 可直接使用的虚拟环境
```conda activate scl```


#### 模型使用
建议使用服务器运行

##### 运行前检查

- 确认```scl.py``` 文件中使用的多GPU编号
  - 执行前先使用```nvidia-smi```命令找到当前可用的GPU
  - 根据可用GPU编号```os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'```,其中```'0,1,2,3'```为运行模型时可以用的GPU编号
- 确认```configs/ ``` 目录下对应文件参数符合需求，其中需要重点关注的配置参数如下
  - train_path 训练文件，做了数据增强后可以修改该路径切换到对应的增强后的文件进行训练
  - valid_path 评估（测试）文件
  - types_path 定义的关系文件
  - train_batch_size 根据GPU数量进行修改，否则可能会爆显存
  - eval_batch_size 
  - neg_entity_count 训练过程产生的最大负样本（实体）数，根据GPU数量进行修改，否则可能会爆显存（100、200、300）
  - neg_relation_count 训练过程产生的最大负样本（关系）数，根据GPU数量进行修改，否则可能会爆显存（100、200、300）
  - train_log_iter  
  - epochs （50、80、100）

##### 运行模型（以当前服务器上的模型为例）
- 进入到配置好的虚拟环境
```shell
conda activate scl
```
- 当需要后台挂起指定输出流时，执行以下命令
```shells
nohup python -u scl.py  > myout.log &  
```
- 查看输出
  - ```myout.log``` 检查程序当前运行状态、运行阶段等
