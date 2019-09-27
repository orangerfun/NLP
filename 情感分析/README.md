### 关于数据
本程序使用IMDB情感分析数据集，IMDB数据集训练集和测试集一共包含25000条已经标注的电影评价。在本程序所在的文件夹中有一个wordlist.npy文件，该文件是事先保存好的数据集中的word列表，里面存储了400000个字，wordVectors.npy是事先训练好的词向量模型，该矩阵中包含400000文本向量，每行50维数据。idsMatrix.npy是保存的索引矩阵。
数据链接：https://pan.baidu.com/s/1M3o7H0am3oantH0KHNuBoA

### 程序介绍
本程序使用了LSTM+一层全链结网络结构，具体实现见程序
