# -*- encode: utf-8 -*-
import re
import codecs
import jieba
import os
from sklearn.model_selection import train_test_split


def split_data(raw_file, ch_file, en_file):
	"""
	将源文本中英文和中文分开，并写入不同文件
	"""
	with codecs.open(raw_file, "r", encoding="utf-8") as fr:
		with codecs.open(ch_file, "w", encoding="utf-8") as fwch:
			with codecs.open(en_file,"w",encoding="utf-8") as fwen:
				for line in fr.readlines():
					ch_line = re.sub(r"^\d+\.", "", line.split("|||")[0].strip())
					ch_line = tokenizer(ch_line, True)

					en_line = re.sub(r"^\d+\.", "", line.split("|||")[1].strip())
					en_line = tokenizer(en_line, False)

					fwch.write(ch_line.strip()+"\n")
					fwen.write(en_line.strip()+"\n")

def tokenizer(line, is_CH=True):
	if is_CH:
		biaodian = r"[\s+、，。？：；‘’”“）（！#￥%……&*\-——《'》\")(/]+"
		line = re.sub(biaodian, "", line)
		seg_list = jieba.cut(line)
		line = " ".join(seg_list)
	else:
		chara = r"[,.?;:\[\])(\"!'*&\^%$#@；]+"
		line = re.sub(chara, " ", line)
	return line

def train_test_split_(ch_data_path, en_data_path, target_path):
	with codecs.open(ch_data_path, "r", encoding="utf-8") as fch:
		with codecs.open(en_data_path, "r", encoding="utf-8") as fen:
			ch_contents, en_contents = fch.readlines(), fen.readlines()
			x_train, x_test, y_train, y_test = train_test_split(ch_contents, en_contents, test_size=0.2)
	if not os.path.exists(target_path):
		os.makedirs(target_path)
	with codecs.open(target_path+"/train_x.txt", "w", encoding="utf-8") as ftrainx:
		with codecs.open(target_path+"/train_y.txt", "w", encoding="utf-8") as ftrainy:
			with codecs.open(target_path+"/test_x.txt", "w", encoding="utf-8") as ftestx:
				with codecs.open(target_path+"/test_y.txt", "w", encoding="utf-8") as ftesty:
					ftrainx.writelines(x_train)
					ftrainy.writelines(y_train)
					ftestx.writelines(x_test)
					ftesty.writelines(y_test)
	print("train and test files have been saved in [%s]"%target_path)


if __name__ == '__main__':
	raw_data_file = "./raw_data/testdata.txt"
	ch_data = "./raw_data/ch_data.txt"
	en_data = "./raw_data/en_data.txt"
	target_path = "./raw_data/train_and_test"

	split_data(raw_data_file, ch_data, en_data)
	train_test_split_(ch_data, en_data, target_path)






