import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from factorGraph import FactorGraph, Message, inference

            
def maxProductFactor(letterboxes, sender, receiver, potential):
    p = gum.Potential(potential)
    senders = []
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    
    p = p.margMaxIn(receiver)
    best_index = p.argmax()[0][receiver]
    p[best_index] = 1
    p[1-best_index] = 0
    
    return Message(sender, receiver, content=p)

def maxProductVariables(letterboxes, sender, receiver, potential):
    p = gum.Potential(potential)
    senders = []
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    
    return Message(sender,receiver,p)

class TreeMaxProductInference:
    def __init__(self,f):
        self.fg = FactorGraph(f)
        self.letterboxes = {}
        for v in f.variables:
            self.letterboxes[v] = []
        for i in range(len(f.factors)):
            self.letterboxes[i] = []

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        self.letterboxes = inference(self, maxProductFactor, maxProductVariables)

    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return {v:self.letterboxes[v][-1].content.argmax()[0][v] for v in self.fg.variables}
    
    def addEvidence(self,dic):
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