import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from factorGraph import FactorGraph, Message, inference

def sumProductFactors(letterboxes, sender, receiver, potential):
    # Fonction de création de message pour les noeuds facteurs, effectue le produit et la somme marginale des messages
    p = gum.Potential(potential)
    senders = []
    
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    content = p.margSumIn(receiver)

    return Message(sender, receiver, content=content)

def sumProductVariables(letterboxes, sender, receiver, potential):
    # Fonction de création de message pour les noeuds variables, effectue le produit des messages
    p = gum.Potential(potential)
    senders = []
    
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    return Message(sender,receiver,p)

class TreeSumProductInference:
    def __init__(self,f):
        self.fg = FactorGraph(f)
        self.letterboxes = {}
        for v in f.variables:
            self.letterboxes[v] = []
        for i in range(len(f.factors)):
            self.letterboxes[i] = []
    
    def makeInference(self):
        """ effectue les calculs de tous les messages """
        self.letterboxes = inference(self, sumProductFactors, sumProductVariables)

    def posterior(self, variable):
        """ retourne la distribution de la variable sous la forme d'un `gum.Potential` """
        try:
            return self.letterboxes[variable][0].content
        except KeyError:
            print("{} not found in variables, try to makeInference on the object.".format(variable))
        except IndexError:
            print("Inference didn't work, maybe because the graph comes from a cyclic Bayesian network.".format(variable))
    
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