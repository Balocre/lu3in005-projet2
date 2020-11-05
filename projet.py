import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python


import utils
from utils import AbstractClassifier # votre code
data=pd.read_csv("heart.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

def getPrior(df):
    """
    Rend un dictionnaire à priori de la classe 1 ainsi que l'intervalle de confiance
    Args : df => fichier
    """
    c=0
    taille = int(df.size/14)
    for n in range(taille):
        d = utils.getNthDict(df,n)
        c += d["target"]
    c = c/(taille)
    return {"estimation":c, "min5pourcent": c - 1.96 * np.sqrt(c*(1-c) / taille) , "max5pourcent": c + 1.96 * np.sqrt(c*(1-c) / taille)}
print(getPrior(train))

class APrioriClassifier(AbstractClassifier):
    def ___init__(self):
        pass
    def estimClass(self,attrs):
        return 1
    def statsOnDF(self,df):
        taille = int(df.size/14)
        VP = 0
        FP = 0
        for n in range(taille):
            d = utils.getNthDict(df,n)
            if d['target'] == 1:
                VP+=1
            else :
                FP+=1
        return {'VP':VP, 'VN':0,'FP':FP,'FN':0,'Précision': VP / (VP+FP), 'Rappel' : 1.0}

cl=APrioriClassifier()
print("test en apprentissage : {}".format(cl.statsOnDF(train)))
print("test en validation: {}".format(cl.statsOnDF(test)))

def P2D_l(df,attr):
    taille = int(df.size/14)
    d1 = dict()
    d0 = dict()
    c0 = 0
    c1 = 0
    for n in range(taille):
        d = utils.getNthDict(df,n)
        if d['target']==1:
            c1 += 1
            if d[attr] in d1:
                d1[d[attr]] += 1
            else:
                d1[d[attr]] = 1
        else :
            c0 += 1
            if d[attr] in d0:
                d0[d[attr]] += 1
            else :
                d0[d[attr] ]= 1
    for k in d1:
        d1[k] /= c1
    for k in d0:
        d0[k] /= c0
    return {1:d1, 0:d0}
print (P2D_l(train,'thal'))

def P2D_p(df, attr):
    taille = int(df.size/14)
    dic = dict()
    for n in range(taille):
        d = utils.getNthDict(df,n)
        if d[attr] not in dic:
                dic[d[attr]] = {1:d['target'], 0:1-d['target']}
        else :
            dic[d[attr]][d['target']]+=1
    for d0 in dic:
        sum = dic[d0][0] + dic[d0][1]
        dic[d0][0] /= sum
        dic[d0][1] /= sum
    return dic
print(P2D_p(train,'thal'))