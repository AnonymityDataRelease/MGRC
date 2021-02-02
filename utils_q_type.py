from utils_msrc import *
import pickle
import string
import re
QTag_W={
    "when":1,
    "where":2,
    "who":3,
    "whose":3,
    "whom":3,
    "who's":3,
    "whio":3,
    "why":4,
    "whi":4,
    "what":5,
    "what's":5,
    "how":6,
    "hoe":6,
    "which":7
}
#"number":8
#"others:0
Time_Key_Words = ["year","year's","years","decades","day","month","decade","era","century","centuries","period","periods","time","times","date","age"]
Place_Key_Words = ["region","religion","city","country","countries","place","area","areas","street","road","position","bay","province","provinces""state","states","river","rivers","park","county","mountain","mountains","locations","location","town","towns"]
Number_Key_Words = ["percent","size","percentage","percentages","amount","number","degrees","degree","accounts"]
Reason_Key_Words = ["reasoning","reason","reasons","because","cause","causes","caused"]


def containsAny(seq, aset):
    seq = seq.split()
    return True if any(i in seq for i in aset) else False
def nextinAny(wtag,seq, alist):
    seq = seq.split()
    word = ""
    for i in range(len(seq)-1):
        if seq[i] == wtag:
            word=seq[i+1]
            break
    return True if (word in alist) else False


def getQTag(question):
    question = re.sub("\?"," ",question.lower())
    tag = []
    question_list = question.split()
    for word,id in QTag_W.items():
        if word in question_list:
            tag.append(id)
    if len(tag) >1:
        tag = tag[0:1]
    if tag == [5]:
        if nextinAny("what",question,Time_Key_Words):
            tag = [1]#time
        if nextinAny("what",question,Number_Key_Words):
            tag = [8]#number
        if nextinAny("what",question,Place_Key_Words):
            tag = [2]#place
        if containsAny(question,Reason_Key_Words):
            tag = [4]
    if tag == [6]:
        if nextinAny("how",question,["many","manys","much","tall","high","far","large","big","small","long","old","deep"]):
            tag = [8]
    if tag == [7]:
        if nextinAny("which",question,Time_Key_Words):
            tag = [1]#time
        if nextinAny("which",question,Place_Key_Words):
            tag = [2]#place
        if containsAny(question,Reason_Key_Words):
            tag = [4]
    if tag == []:
        tag = [0]
    return tag

