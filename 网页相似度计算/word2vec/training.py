# -*- coding: utf-8 -*-
#利用gensim模块训练词向量
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def my_function():
    wiki_news = open('./data/reduce_zhiwiki.txt', 'r',encoding="utf-8")
    '''args: 0:预处理后的语料库；
             1：sg=0表用CBOW模型训练,sg=1表使用skipgram；
             2:windows当前词和预测词可能最大距离；
             3:mincount表最小出现次数，若一个词语出现次数小于min_count,直接忽略该词；
             4：workers使用线程数'''
    model = Word2Vec(LineSentence(wiki_news), sg=0,size=192, window=5, min_count=5, workers=9)
    model.save('zhiwiki_news.word2vec')

if __name__ == '__main__':
    my_function()
