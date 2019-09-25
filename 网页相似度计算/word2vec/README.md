### 程序文件说明
  * data_pre_process.py文件对中文语料进行预处理<br>
  * langconv.py和zh_wiki.py都是预处理中实现繁体中文转化为简体中文的相关程序<br>
  * keyword_extract.py使用jieba实现关键词提取<br>
  * training.py训练词向量，执行后会得到zhiwiki_news系列四个文件<br>
  * word2vec_sim.py计算两个网页的相似度<br>
  * data文件夹存放相关文件，P1/P2.txt是两个要计算相似度的网页文本；P1/P2_keywords.txt是从两个网页中提取出来的关键词；reduce_zhiwiki.txt是通过维基百科语料的txt文本
