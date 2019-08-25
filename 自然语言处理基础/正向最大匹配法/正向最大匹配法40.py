class MM(object):
    def __init__(self):
        self.window_size=3    #最大匹配法中每次词中最大匹配字的个数，也可以从词典中找最长的词的长度

    def cut(self,text):
        index=0
        result=[]
        text_length=len(text)
        dic=["研究生","研究","生命","命","的","起源"]
        while text_length>index:
            for size in range(self.window_size+index,index,-1):
                piece=text[index:size]
                if piece in dic:
                    index=size-1
                    break
            index+=1  #若啥都没匹配到（如标点符号），直接加入结果，并转入下一周期匹配
            result.append(piece)
        return result

if __name__=="__main__":
    text="研究生命的起源,研究起源"
    tokenizer=MM()
    print(tokenizer.cut(text))

