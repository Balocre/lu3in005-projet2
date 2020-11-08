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
#print(getPrior(train))

class APrioriClassifier(AbstractClassifier):
    def ___init__(self):
        pass
    def estimClass(self,attrs):
        return 1
    def statsOnDF(self,df):
        taille = int(df.size/14)
        VP = 0
        VN = 0
        FP = 0
        FN = 0
        for n in range(taille):
            d = utils.getNthDict(df,n)
            estim = self.estimClass(d)
            if estim == 1:
                if d['target'] == 1:
                    VP += 1
                else :
                    FP += 1
            else : 
                if d['target'] == 1:
                    FN += 1
                else:
                    VN += 1
        return {'VP':VP, 'VN': VN,'FP' : FP,'FN' : FN,'Précision': VP / (VP+FP), 'Rappel' : VP / (VP+FN)}

#cl=APrioriClassifier()
#print("test en apprentissage : {}".format(cl.statsOnDF(train)))
#print("test en validation: {}".format(cl.statsOnDF(test)))

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
#print (P2D_l(train,'thal'))

def P2D_p(df, attr):
    dic = dict()
    taille = int(df.size/14)
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
#print(P2D_p(train,'thal'))

class ML2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        self.P2D_l = P2D_l(df,attr)
        self.attr = attr
    def estimClass(self,attrs):
        if (self.P2D_l[0][attrs[self.attr]] > self.P2D_l[1][attrs[self.attr]]) or (np.abs(self.P2D_l[0][attrs[self.attr]] - self.P2D_l[1][attrs[self.attr]]) < 1e-09):
            return 0
        return 1

#cl = ML2DClassifier(train,"thal") # cette ligne appelle projet.P2Dl(train,"thal")
#for i in [0,1,2]:
#    print("Estimation de la classe de l'individu {} par ML2DClassifier : {}".format(i,cl.estimClass(utils.getNthDict(train,i)))) 

#print("test en apprentissage : {}".format(cl.statsOnDF(train)))
#print("test en validation: {}".format(cl.statsOnDF(test)))

class MAP2DClassifier(APrioriClassifier):
    def __init__(self, df, attr):
        self.P2D_p = P2D_p(df, attr)
        self.attr = attr
    def estimClass(self, attrs):
        if (self.P2D_p[attrs[self.attr]][0] > self.P2D_p[attrs[self.attr]][1] or np.abs(self.P2D_p[attrs[self.attr]][0] - self.P2D_p[attrs[self.attr]][1] ) < 1e-09):
            return 0
        return 1


#cl = MAP2DClassifier(train,"thal") # cette ligne appelle projet.P2Dp(train,"thal")
#for i in [0,1,2]:
#    print("Estimation de la classe de l'individu {} par MAP2DClasssifer) : {}".format(i,cl.estimClass(utils.getNthDict(train,i)))) 
#print("test en apprentissage : {}".format(cl.statsOnDF(train)))
#print("test en validation: {}".format(cl.statsOnDF(test)))

def nbParams(df,L = ['target','exang','restecg','ca','trestbps','sex','fbs','cp','age','slope','oldpeak','thalach','chol','thal']):
    #On commence par compter le nombre de valeurs différentes pour chaque attribut
    dic = { i : set() for i in L }
    taille = int(df.size/14)
    for n in range(taille):
        d = utils.getNthDict(df,n)
        for i in L:
            if d[i] not in dic[i]:
                dic[i].add(d[i])
    for k in dic:
        dic[k] = len(dic[k])
    #Ensuite on calcule
    s = 1
    for i in L:
        s *= dic[i]
    s *= 8
    aff = str(len(L)) + " variables(s) : " + str(s) + " octets "
    if s > 1024:
        aff += "= "
        ko = int (s / 1024)
        s = (s % 1024)
        if (ko > 1024):
            mo = int (ko / 1024)
            ko = ko % 1024
            if (mo > 1024):
                go = int(mo / 1024)
                mo = mo % 1024
                aff += str(go) + "go "
            aff += str(mo) +  "mo "
        aff += str(ko) + "ko "
        aff += str(s) + "o"
    print(aff)
    return s

#nbParams(train,['target'])
#nbParams(train,['target','thal'])
#nbParams(train,['target','age'])
#nbParams(train,['target','age','thal','sex','exang'])
#nbParams(train,['target','age','thal','sex','exang','slope','ca','chol'])
#nbParams(train)

def nbParamsIndep(df, L = ['target','exang','restecg','ca','trestbps','sex','fbs','cp','age','slope','oldpeak','thalach','chol','thal']):
    #On commence par compter le nombre de valeurs différentes pour chaque attribut
    dic = { i : set() for i in L }
    taille = int(df.size/14)
    for n in range(taille):
        d = utils.getNthDict(df,n)
        for i in L:
            if d[i] not in dic[i]:
                dic[i].add(d[i])
    for k in dic:
        dic[k] = len(dic[k])
    #Ensuite on calcule
    s = 0
    for i in L:
        s += dic[i]
    s *= 8
    aff = str(len(L)) + " variables(s) : " + str(s) + " octets "
    if s > 1024:
        aff += "= "
        ko = int (s / 1024)
        s = (s % 1024)
        if (ko > 1024):
            mo = int (ko / 1024)
            ko = ko % 1024
            if (mo > 1024):
                go = int(mo / 1024)
                mo = mo % 1024
                aff += str(go) + "go "
            aff += str(mo) +  "mo "
        aff += str(ko) + "ko "
        aff += str(s) + "o"
    print(aff)
    return s
#nbParamsIndep(train,['target'])
#nbParamsIndep(train,['target','thal'])
#nbParamsIndep(train,['target','age'])
#nbParamsIndep(train,['target','age','thal','sex','exang'])
#nbParamsIndep(train,['target','age','thal','sex','exang','slope','ca','chol'])
#nbParamsIndep(train)

