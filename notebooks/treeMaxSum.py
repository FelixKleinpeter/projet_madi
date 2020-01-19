import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from factorGraph import FactorGraph, Message, inference
from itertools import product

def f_transform(potential,f=np.log):
    # Retourne l'application de la fonction f au potentiel donné
    new_potential = gum.Potential(potential)
    p = np.array(new_potential.toarray())
    shpe = p.shape
    p = np.reshape(p,p.size)
    for i,e in enumerate(p):
        if e == 0 and f == np.log:
            p[i] = -np.inf
        p[i] = f(e)
    p = np.reshape(p,shpe)
    if shpe != (1,):
        items = list(product(*[[0,1]]*len(shpe)))
        for item in items:
            new_potential[item] = p[item]
    else:
        new_potential.fillWith(p[0])
    return new_potential

def maxSumFactors(letterboxes, sender, receiver, potential):
    # Fonction de création de message pour les noeuds facteurs, effectue la somme des log-potentiels pour chaque message et 
    # marginalise selon le maximum 
    p = f_transform(potential,np.log)
    senders = []
    for m in letterboxes[sender]:
        if m.sender not in senders:
            m_transform = f_transform(m.content,np.log)
            p = p + m_transform
            senders.append(m.sender)
    p = f_transform(p,np.exp)
    p = p.margMaxIn(receiver)
    best_index = p.argmax()[0][receiver]
    p[best_index] = 1
    p[1-best_index] = 0
    
    return Message(sender, receiver, content=p)

def maxSumVariables(letterboxes, sender, receiver, potential):
    # Fonction de création de message pour les noeuds variables, effectue le produit des messages
    p = gum.Potential(potential)
    senders = []
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    
    return Message(sender,receiver,p)

class TreeMaxSumInference:
    def __init__(self,f):
        self.fg = FactorGraph(f)
        self.letterboxes = {}
        for v in f.variables:
            self.letterboxes[v] = []
        for i in range(len(f.factors)):
            self.letterboxes[i] = []

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        self.letterboxes = inference(self, maxSumFactors, maxSumVariables)

    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return {v:self.letterboxes[v][-1].content.argmax()[0][v] for v in self.fg.variables}
    
    def addEvidence(self,dic):
        # Fonction d'ajout d'évidence dans une instance d'inférence
        for variable,value in dic.items():
            # Si le noeud est une feuille, on change sa valeur
            is_leave = False
            for p in self.fg.factors:
                if len(p.toarray().shape) == 1 and p.var_names[0] == variable:
                    p[value] = 1
                    p[1-value] = 0
                    p = p.normalizeAsCPT()
                    is_leave = True
            # Sinon, on ajoute un noeud facteur relié à la variable
            if not is_leave:
                p = gum.Potential().add(self.fg.variables[variable])
                p[value] = 1
                p = p.normalizeAsCPT()
                self.fg.addFactor(p)
                self.letterboxes[len(self.fg.factors)-1] = []