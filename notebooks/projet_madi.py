import pydotplus as dot
from IPython.display import SVG
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from itertools import product

def factor_repr(f):
    string = "p"+f.var_names[-1]
    if len(f.var_names) > 1:
        string += "g"
        for vn in f.var_names[:-1]:
            string += vn
    return string

class FactorGraph:
    """
    variables : Une dictionaire de nom:gum.DiscreteVariable
    factors : Une liste de gum.Potential
    edges : Une liste de couples (nom_var, id_potential)
    """
    def __init__(self):
        self.variables = {}
        self.factors = []
        self.edges = []
    
    def addVariable(self,v):
        """ Ajout de variable sous la forme de gum.DiscreteVariable """
        self.variables[v.name()] = v
    
    def addFactor(self,p):
        """ Ajout de facteurs sous la forme de gum.Potential """
        self.factors.append(p)
    
    def build(self,bn):
        """ Construit un factor graph à partir d'un gum.BayesNet """
        self.__init__()
        for i in bn.nodes():
            self.addVariable(bn.variable(i))
            self.addFactor(bn.cpt(i))
            for v in bn.cpt(i).var_names:
                self.edges.append((v,i))

    def show(self):
        """ Affichage d'un factor graph """
        string = """
        graph FG {
            layout=neato;
            node [shape=rectangle,margin=0.04,
                  width=0,height=0, style=filled,color="coral"];
            """
        for v in self.variables:
            string += v + ";"
        string += """
            node [shape=point];
            """
        for f in self.factors:
            string += factor_repr(f) + ";"
            
        string += """
            edge;
            """
        
        for v_name,f_id in self.edges:
            string += factor_repr(self.factors[f_id]) + "--" + v_name + ";\n"

        string += "}"
        g=dot.graph_from_dot_data(string)
        return SVG(g.create_svg())
    
    
    def neighbours(self,elm):
        if type(elm) == int:
            return [v_name for (v_name,f_id) in self.edges if f_id == elm]
        elif type(elm) == str:
            return [f_id for (v_name,f_id) in self.edges if v_name == elm]
    
    def leaves(self):
        return [f_id for (_,f_id) in self.edges if len(self.neighbours(f_id)) == 1]
    
    def shortest_path(self,starting_node,ending_node):
        to_visit = self.neighbours(starting_node)
        distances = {node:(starting_node,1) for node in to_visit}
        while len(to_visit) > 0:
            next_gen = []
            for node in to_visit:
                for neigh in self.neighbours(node):
                    if (not neigh in distances.keys()) or distances[neigh][1] > distances[node][1] + 1:
                        distances[neigh] = (node,distances[node][1] + 1)
                        if neigh == ending_node:
                            sequence = [neigh]
                            while sequence[-1] != starting_node:
                                sequence.append(distances[sequence[-1]][0])
                            sequence.reverse()
                            return sequence
                            
                        next_gen.append(neigh)
            to_visit = next_gen
        return None
                    
class Message:
    def __init__(self, sender, receiver, content=[]):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        
    def send(self, letterboxes):
        letterboxes[self.receiver].append(self)

def inference(instance, function):
    # Initialisation
    letterboxes = instance.letterboxes.copy()
    root = np.random.randint(len(instance.fg.factors))
    leaves = instance.fg.leaves()
    if root in leaves:
        leaves.remove(root)
    paths = [instance.fg.shortest_path(leaf,root) for leaf in leaves]
    
    # ETAPE I
    var_step = False
    while sum(len(path) for path in paths) > 0:
        for path in paths:
            if len(path) > 0:
                node = path.pop(0)
                if var_step and node != root:
                    p = gum.Potential()
                    senders = []
                    for m in letterboxes[node]:
                        if m.sender != node and m.sender not in senders:
                            p = p * m.content
                            senders.append(m.sender)
                    message = Message(node,path[0],p)
                    message.send(letterboxes)
                elif not var_step and node != root:
                    message = function(letterboxes,node,path[0],instance.fg.factors[node])
                    message.send(letterboxes)

        var_step = not var_step

    # ETAPE II
    to_visit = [root]
    visited = []

    var_step = False
    while len(to_visit) > 0:
        next_gen = []
        for node in to_visit:
            for neigh in [n for n in instance.fg.neighbours(node) if not n in visited]:
                visited.append(neigh)
                next_gen.append(neigh)
                if var_step:
                    p = gum.Potential()
                    senders = []
                    for m in letterboxes[node]:
                        if m.sender != neigh and m.sender not in senders:
                            p = p * m.content
                            senders.append(m.sender)
                    message = Message(node,neigh,p)
                    message.send(letterboxes)
                else:
                    message = function(letterboxes,node,neigh,instance.fg.factors[node])
                    message.send(letterboxes)
        var_step = not var_step
        to_visit = next_gen

    return letterboxes

def sumProduct(letterboxes, sender, receiver, potential):
    p = gum.Potential(potential)
    senders = []
    for m in letterboxes[sender]:
        if m.sender != receiver and m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    content = p.margSumIn(receiver)

    return Message(sender, receiver, content=content)

class TreeSumProductInference:
    def __init__(self,f):
        self.fg = f
        self.letterboxes = {}
        for v in f.variables:
            self.letterboxes[v] = []
        for i in range(len(f.factors)):
            self.letterboxes[i] = []
    
    def makeInference(self):
        """ effectue les calculs de tous les messages """
        self.letterboxes = inference(self, sumProduct)

    def posterior(self, variable):
        """ retourne la distribution de la variable sous la forme d'un `gum.Potential` """
        try:
            messages_received = self.letterboxes[variable]
            for m in messages_received:
                if m.content[0] < 0.999 and m.content[0] > 0.001:
                    return m.content
        except KeyError:
            print("{} not found in variables, try to makeInference on the object.".format(variable))
            
def maxProduct(letterboxes, sender, receiver, potential):
    p = gum.Potential(potential)
    for m in letterboxes[sender]:
        p = p * m.content
    p = p.margMaxIn(receiver)
    best_index = p.argmax()[0][receiver]
    p[best_index] = 1
    p[1-best_index] = 0
    
    return Message(sender, receiver, content=p)

class TreeMaxProductInference:
    def __init__(self,f):
        self.fg = f
        self.letterboxes = {}
        for v in f.variables:
            self.letterboxes[v] = []
        for i in range(len(f.factors)):
            self.letterboxes[i] = []

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        self.letterboxes = inference(self, maxProduct)

    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return {v:self.letterboxes[v][-1].content.argmax()[0][v] for v in self.fg.variables}

def f_transform(potential,f=np.log):
    new_potential = gum.Potential(potential)
    p = np.array(new_potential.toarray())
    shpe = p.shape
    p = np.reshape(p,p.size)
    for i,e in enumerate(p):
        if e == 0 and f == np.log:
            p[i] = -np.inf
        p[i] = f(e)
    p = np.reshape(p,shpe)
    items = list(product(*[[0,1]]*len(shpe)))
    for item in items:
        new_potential[item] = p[item]
    return new_potential

def maxSum(letterboxes, sender, receiver, potential):
    p = f_transform(potential,np.log)
    for m in letterboxes[sender]:
        m_transform = f_transform(m.content,np.log)
        p = p + m_transform
    p = f_transform(p,np.exp)
    p = p.margMaxIn(receiver)
    best_index = p.argmax()[0][receiver]
    p[best_index] = 1
    p[1-best_index] = 0
    
    return Message(sender, receiver, content=p)

class TreeMaxSumInference:
    def __init__(self,f):
        self.fg = f
        self.letterboxes = {}
        for v in f.variables:
            self.letterboxes[v] = []
        for i in range(len(f.factors)):
            self.letterboxes[i] = []

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        self.letterboxes = inference(self, maxSum)

    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return {v:self.letterboxes[v][-1].content.argmax()[0][v] for v in self.fg.variables}