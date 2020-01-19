import pydotplus as dot
from IPython.display import SVG
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from itertools import product

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
            node [shape=circle,margin=0.04,
                  width=0,height=0, style=filled,color="burlywood"];
            """
        for v in self.variables:
            string += v + ";"
        string += """
            node [shape=rectangle,margin=0.04,
                  width=0,height=0, style=filled,color="coral"];
            """
        for i,f in enumerate(self.factors):
            string += str(i) + ";"
            
        string += """
            edge;
            """
        
        for v_name,f_id in self.edges:
            string += str(f_id) + "--" + v_name + ";\n"

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
    
    def topo_order_in(self, root):
        order = []
        order += self.leaves()
        to_visit = self.leaves()

        while 1:
            new_gen = []
            for node in to_visit:
                if type(node) == str:
                    if root in self.neighbours(node):
                        return order + [root]
                    avialable_neighbours = [n for n in self.neighbours(node) if not n in order]
                    order += avialable_neighbours
                    new_gen += avialable_neighbours
                elif type(node) == int:
                    for neigh in [n for n in self.neighbours(node) if not n in order]:
                        insert = True
                        for parent in self.neighbours(node):
                            if not parent in order and parent != neigh:
                                insert = False
                        if insert and neigh == root:
                            return order + [root]
                        if insert:
                            order.append(neigh)
                            new_gen.append(neigh)

            to_visit = new_gen
    
        return []
    
    def topo_order_out(self,root,topo_in):
        order = []
        order += [root]
        to_visit = [root]

        while len(to_visit) > 0:
            new_gen = []
            for node in to_visit:
                if type(node) == str:
                    avialable_neighbours = [n for n in self.neighbours(node) if not n in order]
                    order += avialable_neighbours
                    new_gen += avialable_neighbours
                elif type(node) == int:
                    for neigh in [n for n in self.neighbours(node) if not n in order]:
                        order.append(neigh)
                        new_gen.append(neigh)

            to_visit = new_gen

        return order
                    
class Message:
    def __init__(self, sender, receiver, content=[]):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        
    def send(self, letterboxes):
        if not self.content in [m.content for m in letterboxes[self.receiver]]:
            letterboxes[self.receiver].append(self)
        
    def __repr__(self):
        return "Sender {}, Receiver {}, Content {}".format(self.sender, self.receiver, self.content)


def inference(instance, functionFactor, functionVariables):
    # Initialisation
    letterboxes = instance.letterboxes.copy()
    leaves = instance.fg.leaves()
    root = np.random.randint(len(instance.fg.factors))
    while root in leaves:
        root = np.random.randint(len(instance.fg.factors))
    order_in = instance.fg.topo_order_in(root)
    
    # ETAPE I
    """
    while sum(len(path) for path in paths) > 0:
        for path in paths:
            if len(path) > 0:
                node = path.pop(0)
                if type(node) == str and node != root:
                    message = functionVariables(letterboxes,node,path[0],gum.Potential())
                    message.send(letterboxes)
                elif type(node) == int and node != root:
                    message = functionFactor(letterboxes,node,path[0],instance.fg.factors[node])
                    message.send(letterboxes)
    """
    visited = []
    for node in order_in:
        visited.append(node)
        for neigh in [n for n in instance.fg.neighbours(node) if not n in visited]:
            if type(node) == str:
                message = functionVariables(letterboxes,node,neigh,gum.Potential())
            elif type(node) == int:
                message = functionFactor(letterboxes,node,neigh,instance.fg.factors[node])
            message.send(letterboxes)
    

    # ETAPE II
    """
    to_visit = [root]
    visited = []

    while len(to_visit) > 0:
        next_gen = []
        for node in to_visit:
            for neigh in [n for n in instance.fg.neighbours(node) if not n in visited]: # AND n in path
                visited.append(neigh)
                next_gen.append(neigh)
                if type(node) == str:
                    message = functionVariables(letterboxes,node,neigh,gum.Potential())
                    message.send(letterboxes)
                elif type(node) == int:
                    message = functionFactor(letterboxes,node,neigh,instance.fg.factors[node])
                    message.send(letterboxes)
        to_visit = next_gen"""
    
    order_out = instance.fg.topo_order_out(root,order_in)
    visited = []
    
    while len(order_out) > 0:
        node = order_out.pop(0)
        visited.append(node)
        
        if type(node) == int and len(letterboxes[node])+1 < len(instance.fg.factors[node].var_names):
            visited.remove(node)
            order_out.insert(1,node)
        else:
            for neigh in [n for n in instance.fg.neighbours(node) if not n in visited]:
                if type(node) == str:
                    message = functionVariables(letterboxes,node,neigh,gum.Potential())
                elif type(node) == int:
                    message = functionFactor(letterboxes,node,neigh,instance.fg.factors[node])
                message.send(letterboxes)
    

    return letterboxes

def sumProductFactors(letterboxes, sender, receiver, potential):
    
    p = gum.Potential(potential)
    senders = []
    
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    content = p.margSumIn(receiver)

    return Message(sender, receiver, content=content)

def sumProductVariables(letterboxes, sender, receiver, potential):
    p = gum.Potential(potential)
    senders = []
    
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    return Message(sender,receiver,p)

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
        self.letterboxes = inference(self, sumProductFactors, sumProductVariables)

    def posterior(self, variable):
        """ retourne la distribution de la variable sous la forme d'un `gum.Potential` """
        try:
            print(self.letterboxes[variable])
            messages_received = self.letterboxes[variable]
            for m in messages_received:
                if m.content[0] < 0.999 and m.content[0] > 0.001:
                    return m.content
        except KeyError:
            print("{} not found in variables, try to makeInference on the object.".format(variable))
            
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
        self.fg = f
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
    if shpe != (1,):
        items = list(product(*[[0,1]]*len(shpe)))
        for item in items:
            new_potential[item] = p[item]
    else:
        new_potential.fillWith(p[0])
    return new_potential

def maxSumFactors(letterboxes, sender, receiver, potential):
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
    p = gum.Potential(potential)
    senders = []
    for m in letterboxes[sender]:
        if m.sender not in senders:
            p = p * m.content
            senders.append(m.sender)
    
    return Message(sender,receiver,p)

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
        self.letterboxes = inference(self, maxSumFactors, maxSumVariables)

    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return {v:self.letterboxes[v][-1].content.argmax()[0][v] for v in self.fg.variables}