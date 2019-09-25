# -*- coding: utf-8 -*-
from gensim.corpora import WikiCorpus
import jieba
from langconv import *
#对中文语料预处理；将xml格式的维基中文语料转换成相应的txt格式
def my_function():
    space = ' '
    i = 0
    l = []
    zhwiki_name = './data/zhwiki-latest-pages-articles.xml.bz2'
    f = open('./data/reduce_zhiwiki1.txt', 'w', encoding="utf-8")
#从xml文件中读出训练语料
    wiki = WikiCorpus(zhwiki_name, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        print("text:",text)
        for temp_sentence in text:
            #将语料中繁体中文转换成简体中文
            temp_sentence = Converter('zh-hans').convert("歐幾里得")
            seg_list = list(jieba.cut(temp_sentence))
            for temp_term in seg_list:
                l.append(temp_term)
        f.write(space.join(l) + '\n')
        l = []
        i = i + 1

        if (i %200 == 0):
            print('Saved ' + str(i) + ' articles')
    f.close()

if __name__ == '__main__':
    my_function()
