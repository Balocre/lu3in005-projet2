# from typing import OrderedDict
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib.pyplot as plt
import utils
import time
from scipy.stats import chi2_contingency
from utils import AbstractClassifier, getNthDict # votre code
data=pd.read_csv("heart.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

def getPrior(df):
    """
    Rend un dictionnaire à priori de la classe 1 ainsi que l'intervalle de confiance.
    param : dataframe à tester
    :return: dict('estimation','moyenne','max5pourcent)
    """
    c=0
    taille = int(df.size/len(df.keys()))
    for n in range(taille):
        d = utils.getNthDict(df,n)
        c += d["target"]
    c = c/(taille)
    return {"estimation":c, "min5pourcent": c - 1.96 * np.sqrt(c*(1-c) / taille) , "max5pourcent": c + 1.96 * np.sqrt(c*(1-c) / taille)}

class APrioriClassifier(AbstractClassifier):
    def ___init__(self):
        """
        Constructeur sans argument. Ne fait rien.
        """
        pass
    def estimClass(self,attrs):
        """
        Etant donné que nous ne pouvons pas mettre en argument le dataframe d'apprentissage, on ne peut pas calculer la moyenne. 
        On estime la classe à partir de ce que l'on sait. Majoritairement, target vaut 1 donc on retourne 1.
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: 1
        """
        return 1
    def statsOnDF(self,df):
        """
        Effectue des statistiques sur le dataframe df pour estimer les VP, VN, FP et FN et calculer la précision et le rappel.
        param : dataframe à tester
        :return: dict('VP','VN','FP','FN', 'Précision', 'Rappel')
        """
        VP = 0
        VN = 0
        FP = 0
        FN = 0
        for n in range(int(df.size/len(df.keys()))):
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
    """
    Calcule dans le dataframe la probabilité P(attr|target) sous la forme d'un dictionnaire asssociant à la valeur t un dictionnaire associant à la valeur a la probabilité P(attr=a|target=t).
    :params: df => dataframe à tester
             attr => attribut à tester
    :return: dict(t : a : P(attr = a|target = t))
    """
    return pd.crosstab(df[attr], df['target'], normalize = 'columns').to_dict() #P(attr|target)


def P2D_p(df, attr):
    """
    Calcule dans le dataframe la probabilité P(target|attr) sous la forme d'un dictionnaire asssociant à la valeur t un dictionnaire associant à la valeur a la probabilité P(target=t|attr=a).
    :params: df => dataframe à tester
             attr => attribut à tester
    :return: dict(a: t: P(target = t|attr = a))
    """
    return pd.crosstab(df['target'], df[attr], normalize = 'columns').to_dict()
class ML2DClassifier(APrioriClassifier):
    #Classifier utilisant notre P2D_l
    def __init__(self, df, attr):
        """
        Constructeur de la classe ML2DClassifier, construit un dictionnaire P(attr = a | target = t)
        :params: df => dataframe à tester
                 attr => attribut à tester
        """
        self.P2D_l = P2D_l(df,attr)
        self.attr = attr
    def estimClass(self,attrs):
        """
        A partir d'un dictionanire d'attributs, estime la classe 0 ou 1 en utilisant les probabilité P(attr|target)
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        #Si P(a|0) est supérieur ou égal à P(a|1) => return 0
        if (self.P2D_l[0][attrs[self.attr]] > self.P2D_l[1][attrs[self.attr]]) or (np.abs(self.P2D_l[0][attrs[self.attr]] - self.P2D_l[1][attrs[self.attr]]) < 1e-20):
            return 0
        #sinon 1
        return 1

class MAP2DClassifier(APrioriClassifier):
    #Classifier utilisant notre P2D_p
    def __init__(self, df, attr):
        """
        Constructeur de la classe MAP2DClassifier, construit un dictionnaire P(target = t | attr = a)
        :params: df => dataframe à tester
                 attr => attribut à tester
        """
        self.P2D_p = P2D_p(df, attr)
        self.attr = attr
    def estimClass(self, attrs):
        #Si P(0|a) est supérieur ou égal à P(1|a) => return 0
        if (self.P2D_p[attrs[self.attr]][0] > self.P2D_p[attrs[self.attr]][1] or np.abs(self.P2D_p[attrs[self.attr]][0] - self.P2D_p[attrs[self.attr]][1] ) < 1e-15):
            return 0
        return 1

def nbParams(df,L = ['target','exang','restecg','ca','trestbps','sex','fbs','cp','age','slope','oldpeak','thalach','chol','thal']):
    """
        Calcule la taille mémoire de ces tables P(target|attr_1,..,attr_k) étant donné un dataframe et la liste [target,attr_1,...,attr_l] en supposant qu'un float est représenté sur 8octets.
        :params: df => dataframe à étudier
                 L => Listre d'attributs
        :return: str : taille mémoire 
    """
    #On commence par compter le nombre de valeurs différentes pour chaque attribut
    dic = { i : set() for i in L } #Dictionnaire qui sera construit ainsi : {attr, ensemble_de_valeurs_differentes}
    for n in range(int(df.size/len(df.keys()))):
        d = utils.getNthDict(df,n)
        for i in L:
            if d[i] not in dic[i]:
                dic[i].add(d[i])
    for k in dic:
        dic[k] = len(dic[k]) #Le dictionnaire devient {attr, nb_valeurs_différentes}
    #Ensuite on calcule
    s = 1
    for i in L:
        s *= dic[i]
    s *= 8
    #formatage de la chaine
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
    """
    Calcule la taille mémoire de ces tables P(target|attr_1,..,attr_k) étant donné un dataframe découpé selon une liste d'attributs supposés indépendants [target,attr_1,...,attr_l] en supposant qu'un float est représenté sur 8octets.
        :params: df => dataframe à étudier
        :return: str : taille mémoire 
    """

    #On commence par compter le nombre de valeurs différentes pour chaque attribut (même chose que précédemment)
    dic = getNthDict(df,0)
    for k in dic :
        dic[k] = set()
    for n in range(int(df.size/len(df.keys()))):
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
    #formatage
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
    """
    A partir d'un dataframe et du nom de la colonne qui est la classe, dessine le graphe.
    :params: df => dataframe à étudier
             classe => classe à étudier
    :return: l'image représentant le graphe 
    """
    d = getNthDict(df,0)
    res = ""
    for i in d:
        if i != classe:
            res+=classe+"->"+i+";"
    return utils.drawGraph(res)

def nbParamsNaiveBayes(df, classe, L = ['target','exang','restecg','ca','trestbps','sex','fbs','cp','age','slope','oldpeak','thalach','chol','thal']):
    """
        calcule la taille mémoire nécessaire pour représenter les tables de probabilité étant donné un dataframe, en supposant qu'un float est représenté sur 8octets et en utilisant l'hypothèse du Naive Bayes.
    """
    #On commence par compter le nombre de valeurs différentes pour chaque attribut (même chose que précédemment)
    if classe not in L:
        L.append(classe)
    dic = { i : set() for i in L }
    for n in range(int(df.size/len(df.keys()))):
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
    #formatage du resultat
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

class MLNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        """
        Constructeur de la classe MLNaiveBayesClassifier, construit un dictionnaire P(attr = a | target = t) pour chaque attribut
        :params: df => dataframe à tester
                 attr => attribut à tester
        """
        self.dic = dict() 
        for k in df.keys():
            if k != "target":
                self.dic[k] = P2D_l(df,k) #Pour chaque attribut, on construit un dictionnaire de probabilité P(attr = a| target = t)

    def estimProbas(self, attrs):
        """
            Estime, pour un dictionnaire d'attributs attrs, la probabilité de chaque classe à partir de l'hypothèse du Naive Bayes.
            :param: attrs => dictionnaire d'attribut
            :return: {0 : P(attr1|0) * P(attr2|0) * P(attr3|0)... , 1 : P(attr1|1) * P(attr2|1) * P(attr3|1)... }
        """
        res0 = 1
        res1 = 1
        for i in attrs.keys():
            if i != "target":
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    #Si la valeur n'est pas dans notre dictionnaire, la probabilité passe à 0
                    res1 = 0
                    break
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    #Si la valeur n'est pas dans notre dictionnaire, la probabilité passe à 0
                    res0 = 0
                    break
        return {0 : res0, 1 : res1}

    def estimClass(self, attrs):
        """
        A partir d'un dictionanire d'attributs, estime la classe 0 ou 1 en utilisant notre estimProbas
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        d = self.estimProbas(attrs)
        if d[0] > d[1] or np.abs(d[0]-d[1]) < 1e-15:
            return 0
        return 1

class MAPNaiveBayesClassifier(APrioriClassifier):
    def __init__(self,df):
        self.df = df
        self.dic = dict() 
        self.pt = getPrior(self.df)['estimation']   #p(target = 1)
        for k in df.keys():
            if k != "target":
                self.dic[k] = P2D_l(df,k)           #Pour chaque attribut, on construit un dictionnaire de probabilité P(attr = a| target = t)

    def estimProbas(self, attrs):
        """
            Estime, pour un dictionnaire d'attributs attrs, la probabilité de chaque classe à partir de l'hypothèse du Naive Bayes.
            :param: attrs => dictionnaire d'attribut
            :return: {0 : P(target = 0) * P(attr1|0) * P(attr2|0) * P(attr3|0)... / rapport_des_2_proba    , 1 : P(target = 1) * P(attr1|1) * P(attr2|1) * P(attr3|1) / rapport_des_2_proba }
        """
        res1 = self.pt      #p(target = 1)
        res0 = 1 - res1     #p(target = 0)
        for i in attrs:
            if i != "target":
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    res1 = 0
                    break
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    res0 = 0
                    break
        if res0 > 1e-15 or res1 > 1e-15:
            return {0 : res0/(res0+res1), 1 : res1/(res0+res1)}
        return {0 : 0, 1 : 0}

    def estimClass(self, attrs):
        """
        A partir d'un dictionanire d'attributs, estime la classe 0 ou 1 en utilisant notre estimProbas
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        d = self.estimProbas(attrs)
        if d[0] > d[1] or np.abs(d[0]-d[1]) < 1e-15:
            return 0
        return 1


def isIndepFromTarget(df,attr,seuil):
    """
        vérifie si `attr` est indépendant de `target` au seuil de x%.
        :params: attr => attribut étudié
                 seuil => float, seuil en pourcentage
        :return: boolean => True si 'attr' est indep de 'target' au seuil de x% sinon False
    """
    cont = pd.crosstab(df['target'],df[attr])
    s,p,dof,exp = chi2_contingency(cont)
    return p > seuil

class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):
    #Même chose que MLNaiveBayesClassifier en ne considérant pas les individus indépendants
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
    #Même chose que MAPNaiveBayesClassifier en ne considérant pas les individus indépendants
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
    """
        A partir d'un dictionnaire dic de {nom:instance de classifier} et d'un dataframe df, représente graphiquement ces classifiers dans l'espace (précision,rappel).
        :params: dic => dic de {nom:instance de classifier}
                 df => dataframe à étudier
        return: image d'un graphique de ces classifier dans l'espace (précision, rappel)
    """
    x=[]    #liste précision
    y=[]    #liste rappel
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
    """
        Calcule l'information mutuelle I(x;y) dans le dataframe df étudié
        :params: df => dataframe étudié
                 x => attribut
                 y => attribut
        :return: Information mutuelle de x et y.
    """
    #On commence par construire 3 tableaux de contingence sous la forme de dictionnaire
    px = dict()     #p(x)
    py = dict()     #p(y)
    pxy = dict()    #p(x,y)
    taille = int(df.size/len(df.keys()))
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
    #On normalise ces tableaux de contingence
    for i in px:
        px[i] /= taille
    for i in py:
        py[i] /= taille
    for i in pxy:
        pxy[i] /= taille
    #On calcule selon la formule donnée
    s=0
    for i in px:
        for j in py:
            if (i,j) in pxy:
                s += pxy[i,j] * np.log2(pxy[i,j]/(px[i]*py[j]))
    return s

def ConditionalMutualInformation(df,x,y,z):
    """
        Calcule l'information mutuelle conditionnelle I(x;y|z) dans le dataframe df étudié
        :params: df => dataframe étudié
                 x => attribut
                 y => attribut
                 z => attribut
        :return: I(x;y|z)
    """
    #On commence par construire 4 tableaux de contingence sous la forme de dictionnaire
    px = set()
    py = set()
    pz = dict()
    pxyz = dict()
    #On parcourt le dataframe
    taille = int(df.size/len(df.keys()))
    for i in range(taille):
        d = getNthDict(df,i)
        if d[x] not in px:
            px.add(d[x])
        if d[y] not in py:
            py.add(d[y])
        if d[z] not in pz:
            pz[d[z]] = 1
        else :
            pz[d[z]] += 1
        if (d[x],d[y],d[z]) not in pxyz:
            pxyz[(d[x],d[y],d[z])] = 1
        else :
            pxyz[(d[x],d[y],d[z])] += 1
    #On normalise nos tableaux de contingence
    for i in pz:
        pz[i] /= taille
    for i in pxyz:
        pxyz[i] /= taille
    #On calcule selon la formule
    s=0
    for i in px:
        for j in py:
            for k in pz:
                if (i,j,k) in pxyz:
                    pxz = 0 #calcul de p(x,z)
                    pyz = 0 #calcul de p(y,z)
                    for (a,b,c) in pxyz:
                        if a == i and c == k:
                            pxz += pxyz[(a,b,c)]
                        if b == j and c == k:
                            pyz += pxyz[(a,b,c)]
                    #calcul de la formule
                    s += pxyz[(i,j,k)] * np.log2(pz[k]*pxyz[(i,j,k)]/(pxz*pyz))
    return s

def MeanForSymetricWeights(a):
    """
    Calcule la moyenne des poids pour une matrice a.
    :params: a => matrice des poids
    :return: moyenne des poids
    """
    s = 0
    (n,m) = np.shape(a)
    for i in range(n):
        for j in range(m):
            s += a[i][j]
    s /= (n*(m-1))
    return s

def SimplifyConditionalMutualInformationMatrix(a):
    """
        annule toutes les valeurs plus petites que cette moyenne dans une matrice  `a` symétrique de diagonale nulle.
        :params: a => matrice des poids
        :return: None
    """
    mean = MeanForSymetricWeights(a)
    (n,m) = np.shape(a)
    for i in range(n):
        for j in range(m):
            if a[i][j] < mean:
                a[i][j] = 0
    return None


def Kruskal(df,a):
    """
        Propose la liste des arcs (non orientés pour l'instant) à ajouter dans notre classifieur sous la forme d'une liste de triplet (attr1,attr2,poids).
        :params: df => dataframe étudié
                 a => matrice des poids
        return : liste des arcs à ajouter sous la forme (attr1,attr2,poids)
    """
    A = []                          #matrice res d'arcs orientés
    noeuds = df.keys()              #liste d'attribut
    union_find=[]                   #ensemble unionfind
    # equivalent wikipedia
        # pour chaque sommet v de G :
            #  créerEnsemble(v)
    for i in noeuds:
        union_find.append({i})
    
    #construction de la matrice d'arrête triees
    AS = []                     #matrice d'arretes triees par ordre décroissant
    (n,m) = np.shape(a)
    for i in range(n):
        for j in range(i,m):
            if a[i][j] > 1e-15:
                AS.append((noeuds[i], noeuds[j], a[i][j]))
    AS.sort(key=lambda tup: tup[2], reverse = True) 

    #equivalent : 
    # pour chaque arête (u, v) de G prise par poids décroissant :
    for (u,v,x) in AS:  
        fu = set() #find(u)
        fv = set() #find(v)
        #calcul de find(u)
        for eu in union_find:
            if u in eu:
                fu = eu
                break
        #calcul de find(v)
        for ev in union_find:
            if v in ev:
                fv = ev
                break
        #equivalent ! si find(u) ≠ find(v) :
        if fu != fv:
            #equivalent :  
                # ajouter l'arête (u, v) à l'ensemble A
                # union(u, v)
            A.append((u,v,x))
            union_find.append(fu.union(fv))
            if fu in union_find :
                union_find.remove(fu)
            if fv in union_find:
                union_find.remove(fv)
    return A

def ConnexSets(L):
    """
        Rend une liste d'ensemble d'attributs connectés.
        :params: liste ('attr_a', 'attr_b', poids)
        :return: liste d'ensembles connectés
    """
    res = [] # liste d'ensembles connectés
    # Pour chaque tuple dans L
    for (i,j,k) in L:
        b = True
        # On regarde si un de ses attributs est dans un ensemble de notre liste res
        for ens in res:
            # Si oui, on rajoute l'autre dans cet ensemble
            if i in ens or j in ens:
                ens.add(j)
                ens.add(i)
                b = False
                break
            # Sinon on crée un ensemble et on met la paire dedans
        if b:
            res.append({i,j})
    return res

def OrientConnexSets(df,L,attr):
    """
        Compare l'information mutuelle des deux attributs par rapport à attr pour l'orienter.
        Cette fonction vérifie qu'un attr ne peut pas avoir plus d'un seul parent en plus de target.
        Si un attribut doit avoir plusieurs parents, il conserve celui de poids maximum par rapport à target
        et devient le parent des autres (on inverse l'orientation des arcs).

        :params: df   => dataframe à étudier
                 L    => liste de (attr_a, attr_b, poids) non orienté
                 attr => attribut étudié (target)
        :return: liste de (attr_a, attr_b) orienté
    """
    res = []    #liste résultat
    s = set()   #Ensemble d'attributs qui ont déjà un parent.
    for (x,y,a) in L:
        if y in s :             #si y a déjà un parent
            if x in s:              #si x aussi
                continue                #on ignore la paire
            res.append((y,x))       #sinon on l'inverse
            s.add(x)
            continue
        if x in s :             #si x a déjà un parent
            if y in s:              #si y aussi
                continue                # on ignore la paire
            res.append((x,y))       #sinon on l'inverse
            s.add(y)
            continue
        #comparaison des informations mutuelle
        if MutualInformation(df,x,attr) > MutualInformation(df,y,attr):
            #ajout de l'arc (x,y)
            s.add(y)
            res.append((x,y))
        else :
            #ajout de l'arc (y,x)
            s.add(x)
            res.append((y,x))
    return res

class MAPTANClassifier(MAPNaiveBayesClassifier):
    def __init__(self, df):
        """
            Constructeur de la classe MAPTANClassifier. Calcule les dictionnaires de probabilités pour chaque attribut à 1 parent et chaque attribut à 2 parents.
            :params: df => dataframe à étudier
        """
        MAPNaiveBayesClassifier.__init__(self, df) #renvoie pt qui vaut P(target = 1) et dic qui contient un p2d_l pour chaque attribut
        #matrice des poids
        cmis=np.array([[0 if x==y else ConditionalMutualInformation(train,x,y,"target") 
                for x in train.keys() if x!="target"]
                for y in train.keys() if y!="target"])
        SimplifyConditionalMutualInformationMatrix(cmis)
        #calcul des arcs
        self.liste_arcs = OrientConnexSets(train, Kruskal(df, cmis), 'target') 
    
        self.dic2 = dict()      # dictionnaire qui pour chaque attribut contient un dictionnaire P(fils|(pere, target)) de forme {attr : (pere,target) : fils : proba}
        self.enfant = set()     # Ensemble d'attr qui ont un pere en plus de target
        for (pere,fils) in self.liste_arcs:
            self.enfant.add(fils)
            self.dic2[(pere,fils)] = self.P_fils_sachant_parent(df,pere,fils)

    def P_fils_sachant_parent(self, df, pere, fils): #P(fils | (pere,target))
        """
            Calcule dans le dataframe la probabilité P(fils = f|(pere = p, target = t)) sous la forme d'un dictionnaire asssociant à la valeur (p,t) un dictionnaire associant à la valeur f la probabilité P(fils = f|(pere = p, target = t)).
            :params: df   => dataframe à tester
                     pere => attribut pere
                     fils => attribut fils
            :return: {(pere = p,target = t) : fils = f : P(fils = f|(pere = p, target = t))}
        """
        # Construction d'une table de contingence sous la forme de dictionnaire {(pere,target) : fils : nb_occurence}
        dic = dict()
        for i in range(int(df.size/len(df.keys()))):
            d = getNthDict(df,i)
            if (d[pere], d['target']) not in dic:
                dic[(d[pere], d['target'])] = dict()
            d2 = dic[(d[pere], d['target'])]
            if d[fils] not in d2:
                d2[d[fils]] = 1
            else :
                d2[d[fils]] += 1
        # Normalisation du dictionnaire sous la forme {(pere,target) : fils : proba}
        for i in dic:
            d = dic[i]
            count = 0
            for j in d :
                count += d[j]
            for j in d :
                d[j] /= count
        return dic

    def estimProbas(self, attrs):
        """

        """
        res1 = self.pt      #P(target = 1)
        res0 = 1 - res1     #P(target = 0)

        #produit P(target) * P(attr1 | target) * P(attr2 | target)... Pour tous ceux qui n'ont qu'un parent
        for i in attrs:
            if i != "target" and i not in self.enfant:
                d = self.dic[i]
                if attrs[i] in d[1]:
                    res1 *= d[1][attrs[i]]
                else :
                    res1 = 0
                    break
                if attrs[i] in d[0]:
                    res0 *= d[0][attrs[i]]
                else :
                    res0 = 0
                    break
        
        #produit P(fils|(pere,target)) pour tous ceux qui ont deux parents
        for (parent,fils) in self.liste_arcs:
            d = self.dic2[(parent,fils)]
            if (attrs[parent], 0) in d:
                if attrs[fils] in d[(attrs[parent],0)]:
                    res0 *= d[(attrs[parent],0)][attrs[fils]]
                else:
                    res0 = 0
            if (attrs[parent], 1) in d:
                if attrs[fils] in d[(attrs[parent],1)]:
                    res1 *= d[(attrs[parent],1)][attrs[fils]]
                else:
                    res1 = 0
            #cas tuple absent du dictionnaire, on repasse à un seul parent (target)
            if (attrs[parent], 0) not in d and (attrs[parent], 1) not in d :
                return {0 : 0, 1 : 0}

        if res0 > 0 or res1 > 0:
            return {0 : res0/(res0+res1), 1 : res1/(res0+res1)}
        return {0 : res0, 1 : res1}

    def draw(self):
        res = ""
        for i in self.df.keys():
            if i != "target":
                res += "target->"+i+";"
        for (a,b) in self.liste_arcs:
                res += a+"->"+b+";"
        return utils.drawGraph(res)