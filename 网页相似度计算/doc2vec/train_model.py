#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim.models as g
from gensim.corpora import WikiCorpus
import logging
from langconv import *

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

docvec_size=192
class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        import jieba
        for content, (page_id, title) in self.wiki.get_texts():
            #输入文件格式：每一行为：<labels, words>， 其中labels 可以有多个，用tab 键分隔，words 用空格键分隔，eg:<id　　category　　I like my cat demon>.
            #输出为词典vocabuary 中每个词的向量表示，这样就可以将商品labels：id，类别的向量拼接用作商品的向量表示
            yield g.doc2vec.LabeledSentence(words=[w for c in content for w in jieba.cut(Converter('zh-hans').convert(c))], tags=[title])

def my_function():
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    documents = TaggedWikiDocument(wiki)

    '''args: docs:训练的语料文章
             size:段落向量的维度
             window：当前词和预测词的最大距离
             min_count:最小出现次数
             workers:训练词向量时使用的线程数
             dm: 训练时使用的模型种类，默认dim=1,即使用DM模型；当dm为其他值时，使用DBOW模型训练词向量
             dbow_words：当设为1时，则在训练doc_vector（DBOW）的同时训练Word_vector（Skip-gram）；默认为0，只训练doc_vector，速度更快'''

    model = g.Doc2Vec(documents, dm=0, dbow_words=1, size=docvec_size, window=8, min_count=19, workers=8)
    model.save('data/zhiwiki_news.doc2vec')

if __name__ == '__main__':
    my_function()

