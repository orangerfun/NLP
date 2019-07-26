import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from nltk import FreqDist
import seaborn as sns
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#打开文件
def fileopen(filepath):
	file=open(filepath,"r",encoding="utf8")
	file_content=file.read()
	file.close()
	return file_content

#统计词频并画图
def freq(review,nlarge=30):
	all_sentence=[sent for sent in review]
	all_sentence=" ".join(all_sentence)
	all_words=all_sentence.split()
	word_freq=FreqDist(all_words)
	df_freqdist=pd.DataFrame({"word":list(word_freq.keys()),"count":list(word_freq.values())})
	d=df_freqdist.nlargest(columns="count",n=nlarge)
	plt.figure(figsize=(20,10))
	# plt.bar(d["word"],d["count"])
	sns.barplot(d["word"],d["count"])
	plt.show()

#去除停用词
def strip_stopword(review):
	not_stopword=[]
	for sentence in review:
		temp=[]
		for word in sentence:
			if nlp.vocab[word].is_stop==True:
				continue
			temp.append(word)
		not_stopword.append(" ".join(temp))
		# print(not_stopword)
	return not_stopword


pos_dir=os.listdir(".//positive")
neg_dir=os.listdir(".//negative")
print(len(pos_dir))
print(len(neg_dir))
pos_content=[fileopen(".//positive/"+filename) for filename in pos_dir]
neg_content=[fileopen(".//negative/"+filename) for filename in neg_dir]

df=pd.DataFrame({"reviews":pos_content+neg_content,"class":0})
df["class"][:len(pos_content)]=1
# print(df["reviews"][4])

replace1=re.compile("(\,)|(\.)|(\')|(\!)|(\?)|(\;)|(\:)|(\()|(\))|(\*)|(\-)|(\")")
# 匹配<br />
replace2=re.compile("<br\s*/>")
reviews=[replace1.sub("",line.lower()) for line in df["reviews"]]
reviews=[replace2.sub("",line) for line in reviews]
df["reviews"]=reviews#[句子]
# print(df["reviews"][15])
# freq(df["reviews"])

#加载英语模型；spacy是自然语言处理的一个库
nlp=spacy.load("en_core_web_sm")
nlp.vocab["however"].is_stop=False
nlp.vocab["but"].is_stop=False
nlp.vocab["no"].is_stop=False
nlp.vocab["not"].is_stop=False
df["tokenizer"]=[line.split() for line in df["reviews"]]

df["cleaned_review"]=strip_stopword(df["tokenizer"])
print(df.columns)

# x_train,x_temp,y_train,y_temp=train_test_split(df["cleaned_review"],df["class"],stratify=df["class"],test_size=0.3)
# x_test,x_val,y_test,y_val=train_test_split(x_temp,y_temp,stratify=y_temp,test_size=0.5)


train,temp=train_test_split(df,stratify=df["class"],test_size=0.3,random_state=41)
''' stratify是为了保持split前类的分布。比如有100个数据，80个属于A类，20个属于B类。如果train_test_split(... test_size=0.25, stratify = y_all), 那么split之后数据如下： 
	training: 75个数据，其中60个属于A类，15个属于B类。 
	testing: 25个数据，其中20个属于A类，5个属于B类。 

	用了stratify参数，training集和testing集的类的比例是 A：B= 4：1，等同于split前的比例（80：20）。通常在这种类分布不平衡的情况下会用到stratify。
	将stratify=X就是按照X中的比例分配 
	将stratify=y就是按照y中的比例分配 '''

test,val=train_test_split(temp,stratify=temp["class"],test_size=0.5,random_state=41)
# print(train.shape,test.shape,val.shape)

#词袋模型
bow=CountVectorizer()
bow_train=bow.fit_transform(train["cleaned_review"])
print("bow:",bow_train)
'''fit_transform：
　  学习词汇字典并转换为矩阵。等同于先调用fit，再调用transform，不过更有效率。
　　返回向量矩阵。[n_samples, n_features]。行表示文档，列表示特征'''
bow_val=bow.transform(val["cleaned_review"])
bow_test=bow.transform(test["cleaned_review"])

#Tfidf模型
tfidf=TfidfVectorizer()
tfidf_train=tfidf.fit_transform(train["cleaned_review"])
tfidf_val=tfidf.transform(val["cleaned_review"])
tfidf_test=tfidf.transform(test["cleaned_review"])

# lr=LogisticRegression()
# lr.fit(bow_train,train["class"])
# pre_val=lr.predict(bow_val)
# pre_test=lr.predict(bow_test)
# print("validation accuracy score:",accuracy_score(val["class"],pre_val))
# print("test accuracy score:",accuracy_score(test["class"],pre_test))

svc=SVC()
svc.fit(tfidf_train,train["class"])
pre_val=svc.predict(tfidf_val)
pre_test=svc.predict(tfidf_test)
print("validation accuracy score:",accuracy_score(val["class"],pre_val))
print("test accuracy score:",accuracy_score(test["class"],pre_test))

