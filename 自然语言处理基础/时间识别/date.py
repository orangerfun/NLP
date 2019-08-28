import re
from datetime import datetime,timedelta
from dateutil.parser import parse
import jieba.posseg as psg

UTIL_CN_NUM={"零":0,"一":1,"二":2,"两":2,"三":3,"四":4,
    "五":5,"六":6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

UTIL_CN_UNIT={'十': 10, '百': 100, '千': 1000, '万': 10000}

#将汉字表示的数字转换成数字
def  cn2dig(src):
    if src=="":
        return None
    m=re.match(r"\d+",src)
    if m:
        return int(m.group(0))
    rsl=0
    unit=1
    #从“个位”读起
    for item in src[::-1]:
        if item in UTIL_CN_UNIT:
            unit=UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM:
            num=UTIL_CN_NUM[item]
            rsl+=num*unit
        else:
            return None
    if rsl<unit:
        rsl+=unit
    return rsl


#将汉字等表示的年份规范化
def year2dig(year):
    res=""
    for item in year:
        if item in UTIL_CN_NUM:
            res=res+str(UTIL_CN_NUM[item])
        else:
            res=res+item
    m=re.match(r"\d+",res)
    if m:
        #当年份是如19年这样简写时将其重新写成2019这样完整的
        if len(m.group(0))==2:
            return int(datetime.datetime.today().year/100)*100+int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


#将时间解析成标准格式
def parse_datetime(msg):
    if msg is None or len(msg)==0:
        return None
#将此行注释:还不如直接匹配法识别时间更准，有些时间parse会解析成错误的时间而不发生Exception,致使下面代码无法继续执行
    # try:
    #     dt=parse(msg,fuzzy=True)   #fuzzy开启模糊匹配，过滤掉无法识别的时间日期字符
    #     return dt.strftime("%Y-%m-%d %H:%M:%S")
    # except Exception as e:
    else:
        m=re.match(r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",msg)
        # print("m:",m)
        #m.group(0)表示所有匹配到的东西;m.group(1)表示第一个分组匹配到的东西，其他依次类推
        if m.group(0) is not None:
            res={
                "year": m.group(1),
                "month": m.group(2),
                "day": m.group(3),
                "hour": m.group(5) if m.group(5) is not None else '00',
                "minute": m.group(6) if m.group(6) is not None else '00',
                "second": m.group(7) if m.group(7) is not None else '00',
            }
            param={}
            for name in res:
                if res[name] is not None and len(res[name])!=0:
                    temp=None
                    if name == "year":
                        temp=year2dig(res[name][:-1])
                    else:
                        temp=cn2dig(res[name][:-1])
                    if temp is not None:
                        param[name]=int(temp)
            target_date=datetime.today().replace(**param)   #replace()方法将指定参数时间修改，未指定的为当前时间
            is_pm=m.group(4)
            if is_pm is not None:
                if is_pm=="下午" or is_pm=="中午" or is_pm=="晚上":
                    hour=target_date.time().hour  #hour属性获取时间中的”时“
                    #换算成24小时制，区分上午下午
                    if hour<12:
                        target_date=target_date.replace(hour=hour+12)
            return target_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return None


#检查提取出的日期是否有效
def check_time_valid(word):
    
    m=re.match(r"\d+$",word)
    if m:
        if len(word)<=6:
            return None

    #将不是以数字 年月日号时分秒结尾的过滤掉        
    if re.search(r"[^\d+号日年月点分时秒]$",word):
        return None

    word1=re.sub(r"[号|日]\d+$","日",word)
    if word1!=word:
        return check_time_valid(word1)
    else:
        return word1
    

#从文本中提取出时间
def time_extract(text):
    time_res=[]
    word=""
    keyDate={"今天":0,"明天":1,"后天":2}
    for k,v in psg.cut(text):
        if k in keyDate:
            if word!="":
                time_res.append(word)
            word=(datetime.today()+timedelta(days=keyDate[k])).strftime("%Y{}%m{}%d{}").format("年","月","日")
        elif word!="":
            if v in ["m","t"]:      #m代表数词，t代表时间词
                word+=k
            else:
                time_res.append(word)
                word=""
        elif v in ["m","t"]:
            word=k
    if word!="":
        time_res.append(word)
    result=list(filter(lambda x: x is not None,[check_time_valid(w) for w in time_res]))
    # print("result:",result)
    final_res=[parse_datetime(w) for w in result]
    return [x for x in final_res if x is not None]


if __name__=="__main__":
    text1='我要住到明天下午三点'
    print(text1,time_extract(text1),sep=":")
    
    text2="预定28号的房间"
    print(text2, time_extract(text2), sep=':')

    text3 = '我要从26号下午4点住到11月2号'
    print(text3, time_extract(text3), sep=':')

    text4 = '我要预订今天到三十号的房间'
    print(text4, time_extract(text4), sep=':')

    text5 = '在20号预留15个房间'
    print(text5, time_extract(text5), sep=':')