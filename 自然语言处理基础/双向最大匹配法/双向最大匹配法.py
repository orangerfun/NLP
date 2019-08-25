class MM_RMM(object):
    def __init__(self):
        self.window_size=3   #最大匹配法中每次词中最大匹配字的个数，也可以从词典中找最长的词的长度
        self.dic=["研究生","研究","生命","命","的","起源"]
        # self.dic=["南京","市长","南京市长","南京市","长江","大桥","长江大桥"]
#正向匹配
    def cut_MM(self,text):
        index=0
        result=[]
        text_length=len(text)
        while text_length>index:
            for size in range(self.window_size+index,index,-1):
                piece=text[index:size]
                if piece in self.dic:
                    index=size-1
                    break
            index+=1  #若啥都没匹配到（如标点符号），直接加入结果，并转入下一周期匹配
            result.append(piece)
        return result
#逆向匹配
    def cut_RMM(self,text):
        result=[]
        index=len(text)
        while index>0:
            for size in range(index-self.window_size,index):
                piece=text[size:index]
                if piece in self.dic:
                    index=size+1
                    break
            index-=1
            result.append(piece)
        result.reverse()
        return result
#双向最大匹配规则
    def final_result(self,result1,result2):
        count1=count2=0
        if len(result1)<len(result2):
            return result1
        elif len(result1)>len(result2):
            return result2
        else:
            if result1==result2:
                return result1
            else:
                for i in result1:
                    if len(i)==1:
                        count1+=1
                for j in result2:
                    if len(j)==1:
                        count2+=1
                if i<j:
                    return result1
                else:
                    return result2

if __name__=="__main__":
    text="研究生命的起源"
    # text="南京市长江大桥"
    tokernizer=MM_RMM()
    print(tokernizer.cut_MM(text))
    print(tokernizer.cut_RMM(text))
    result1,result2=tokernizer.cut_MM(text),tokernizer.cut_RMM(text)
    print(tokernizer.final_result(result1,result2))