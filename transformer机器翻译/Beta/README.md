# 测试版Transformer
本程序首先完成德语到英语的翻译，后进行稍微改变实现中英文互译
# 程序结构
由7个文件构成 <br>
* `clean_data.py`处理原始语料（原始语料格式见test.txt），包括分割数据集，分词，去除标点等
* `heperparams`参数设置，除了calean_data.py文件中参数，其他文件参数都在该文件中定义
* `prepro.py`数据处理，主要是构建此表
* `data_load.py` 加载数据，包括构建词典，加载训练/测试数据
* `modules.py` transformer模型文件，包括LN, Embedding, Position_encoding, Multihead_attention, feedforward等
* `train.py` 程序训练主入口
* `eval.py` 验证程序，使用测试集验证模型效果
# 执行程序
* 1.运行 `clean_data.py`,生成训练集和测试集数据
* 2.运行 `prepro.py`生成词表
* 3.运行 `train.py`训练模型
* 4.运行 `eval.py`使用训练好的模型进行测试，查看模型效果
# 数据说明
写本程序的目的是为了进一步熟悉Transformer，因此并没有在意程序的结果；使用数据集并不大，本次使用的数据集来自某公司，
公司提供的数据集非常大，出于训练时间考虑，本次只选择其中`10 0000`条训练


