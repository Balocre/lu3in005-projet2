import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib.pyplot as plt
import utils
from scipy.stats import chi2_contingency
from utils import AbstractClassifier, getNthDict # votre code
data=pd.read_csv("heart.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

def getPrior(df):
    """
    Rend un dictionnaire à priori de la classe 1 ainsi que l'intervalle de confiance
    Args : df => fichier
    """
    c=0
    taille = int(df.size/len(getNthDict(df,0)))
    for n in range(taille):
        d = utils.getNthDict(df,n)
        c += d["target"]
    c = c/(taille)
    return {"estimation":c, "min5pourcent": c - 1.96 * np.sqrt(c*(1-c) / taille) , "max5pourcent": c + 1.96 * np.sqrt(c*(1-c) / taille)}

class APrioriClassifier(AbstractClassifier):
    def ___init__(self):
        pass
    def estimClass(self,attrs):
        return 1
    def statsOnDF(self,df):
        taille = int(df.size/len(getNthDict(df,0)))
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


def P2D_l(df,attr):
    taille = int(df.size/len(getNthDict(df,0)))
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

def P2D_p(df, attr):
    dic = dict()
    taille = int(df.size/len(getNthDict(df,0)))
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

class ML2DClassifier(APrioriClassifier):

    def __init__(self, df, attr):
        self.P2D_l = P2D_l(df,attr)
        self.attr = attr
    def estimClass(self,attrs):
        if (self.P2D_l[0][attrs[self.attr]] > self.P2D_l[1][attrs[self.attr]]) or (np.abs(self.P2D_l[0][attrs[self.attr]] - self.P2D_l[1][attrs[self.attr]]) < 1e-15):
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
        if (self.P2D_p[attrs[self.attr]][0] > self.P2D_p[attrs[self.attr]][1] or np.abs(self.P2D_p[attrs[self.attr]][0] - self.P2D_p[attrs[self.attr]][1] ) < 1e-15):
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
    taille = int(df.size/len(getNthDict(df,0)))
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


def nbParamsIndep(df):
    #On commence par compter le nombre de valeurs différentes pour chaque attribut
    dic = getNthDict(df,0)
    for k in dic :
        dic[k] = set()
    taille = int(df.size/len(dic))
    for n in range(taille):
        d = utils.getNthDict(df,n)
        for i in d:
            if d[i] not in dic[i]:
                dic[i].add(d[i])
    for k in dic:
        dic[k] = len(dic[k])
    #Ensuite on calcule
    s = 0
    for i in dic:
        s += dic[i]
    s *= 8
    aff = str(len(dic)) + " variables(s) : " + str(s) + " octets "
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

def drawNaiveBayes(df,classe):
    d = getNthDict(df,0)
    res = ""
    for i in d:
        if i != classe:
            res+=classe+"->"+i+";"
    return utils.drawGraph(res)

def nbParamsNaiveBayes(df, classe, L = ['target','exang','restecg','ca','trestbps','sex','fbs','cp','age','slope','oldpeak','thalach','chol','thal']):
    nb_v = len(L)
    if classe not in L:
        L.append(classe)
    dic = { i : set() for i in L }
    taille = int(df.size/len(getNthDict(df,0)))
    for n in range(taille):
        d = utils.getNthDict(df,n)
        for i in L:
            if d[i] not in dic[i]:
                dic[i].add(d[i])
    for k in dic:
        dic[k] = len(dic[k])
    #Ensuite on calcule
    s = dic[classe]*8
    L.remove(classe)
    for i in L:
        s += (dic[i]*dic[classe]) * 8
    aff = str(nb_v) + " variables(s) : " + str(s) + " octets "
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

class MLNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        self.dic = dict() 
        for k in df.keys():
            if k != "target":
                self.dic[k] = P2D_l(df,k)

    def estimProbas(self, attrs):
        res0 = 1
        res1 = 1
        for i in attrs.keys():
            if i != "target":
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    res1 = 0
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    res0 = 0
        return {0 : res0, 1 : res1}

    def estimClass(self, attrs):
        d = self.estimProbas(attrs)
        if d[0] > d[1] or np.abs(d[0]-d[1]) < 1e-15:
            return 0
        return 1

class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        self.df = df
        self.dic = dict() 
        for k in df.keys():
            if k != "target":
                self.dic[k] = P2D_l(df,k)

    def estimProbas(self, attrs):
        #Calcul de P(T)
        p_t = 0
        for k in self.df['target']:
            if k == 1:
                p_t+=1
        p_t /= self.df['target'].size
        #Calcul a posteriori
        res0 = 1-p_t
        res1 = p_t
        for i in attrs:
            if i != "target":
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    res1 = 0
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    res0 = 0
        if res0 > 1e-15 or res1 > 1e-15:
            return {0 : res0/(res0+res1), 1 : res1/(res0+res1)}
        return {0 : 0, 1 : 0}

    def estimClass(self, attrs):
        d = self.estimProbas(attrs)
        if d[0] > d[1] or np.abs(d[0]-d[1]) < 1e-15:
            return 0
        return 1

def isIndepFromTarget(df,attr,seuil):
    cont = pd.crosstab(df['target'],df[attr])
    s,p,dof,exp = chi2_contingency(cont)
    return p > seuil

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    def __init__(self, df, seuil):
        MLNaiveBayesClassifier.__init__(self,df)
        self.seuil = seuil
        self.df = df
        self.indep = set()
        for i in df.keys():
            if isIndepFromTarget(df,i,seuil):
                self.indep.add(i)
    def estimProbas(self, attrs):
        b = attrs.copy() 
        for k in attrs:
            if k in self.indep:
                b.pop(k)
        return MLNaiveBayesClassifier.estimProbas(self,b)

    def draw(self):
        res = ""
        for i in self.df.keys():
            if i != "target":
                if i not in self.indep:
                    res+="target->"+i+";"
        return utils.drawGraph(res)

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df, seuil):
        MAPNaiveBayesClassifier.__init__(self,df)
        self.seuil = seuil
        self.df = df
        self.indep = set()
        for i in df.keys():
            if isIndepFromTarget(df,i,seuil):
                self.indep.add(i)
    def estimProbas(self, attrs):
        b = attrs.copy() 
        for k in attrs:
            if k in self.indep:
                b.pop(k)
        return MAPNaiveBayesClassifier.estimProbas(self,b)

    def draw(self):
        res = ""
        for i in self.df.keys():
            if i != "target":
                if i not in self.indep:
                    res+="target->"+i+";"
        return utils.drawGraph(res)


def mapClassifiers(dic,df):
    x=[]
    y=[]
    for i in dic:
        d = dic[i].statsOnDF(df)
        x.append(d["Précision"])
        y.append(d["Rappel"])

    fig, ax = plt.subplots()
    ax.scatter(x, y, c ='red', marker = 'x')

    for i in range(len(x)):
        ax.annotate(str(i+1),(x[i], y[i]))
    return ax

def MutualInformation(df,x,y):
    px = dict()
    py = dict()
    pxy = dict()
    taille = int(df.size/len(getNthDict(df,0)))
    for i in range(taille):
        d = getNthDict(df,i)
        if d[x] not in px:
            px[d[x]] = 1
        else :
            px[d[x]] += 1
        if d[y] not in py:
            py[d[y]] = 1
        else :
            py[d[y]] += 1
        if (d[x],d[y]) not in pxy:
            pxy[(d[x],d[y])] = 1
        else :
            pxy[(d[x],d[y])] += 1
    for i in px:
        px[i] /= taille
    for i in py:
        py[i] /= taille
    for i in pxy:
        pxy[i] /= taille
    s=0
    for i in px:
        for j in py:
            if (i,j) in pxy:
                s += pxy[i,j] * np.log2(pxy[i,j]/(px[i]*py[j]))
    return s
