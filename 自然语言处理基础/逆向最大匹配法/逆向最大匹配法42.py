class RMM(object):
    def __init__(self):
        self.window_size=3    #最大词长度

    def cut(self,text):
        result=[]
        index=len(text)
        dic=["研究生","研究","生命","命","的","起源"]
        while index>0:
            for size in range(index-self.window_size,index):
                piece=text[size:index]
                if piece in dic:
                    index=size+1
                    break
            index-=1
            result.append(piece)
        result.reverse()
        return result

if __name__=="__main__":
    text="研究生命的起源"
    tokernizer=RMM()
    print(tokernizer.cut(text))

