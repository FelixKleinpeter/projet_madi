import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from itertools import product
import pydotplus as dot
from IPython.display import SVG

class FactorGraph:
    """
    variables : Une dictionaire de nom:gum.DiscreteVariable
    factors : Une liste de gum.Potential
    edges : Une liste de couples (nom_var, id_potential)
    """
    def __init__(self,fg=None):
        if fg == None:
            self.variables = {}
            self.factors = []
            self.edges = []
        else:
            self.variables = {vn:v for vn,v in fg.variables.items()}
            self.factors = [gum.Potential(p) for p in fg.factors]
            self.edges = [e for e in fg.edges]
    
    def addVariable(self,v):
        """ Ajout de variable sous la forme de gum.DiscreteVariable """
        self.variables[v.name()] = v
    
    def addFactor(self,p):
        """ Ajout de facteurs sous la forme de gum.Potential """
        self.factors.append(p)
        for v in p.var_names:
            self.edges.append((v,len(self.factors)-1))
    
    def build(self,bn):
        """ Construit un factor graph à partir d'un gum.BayesNet """
        self.__init__()
        for i in bn.nodes():
            self.addVariable(bn.variable(i))
            self.addFactor(bn.cpt(i))

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
        if len(order) == 0:
            return []

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
    
    def topo_order_out(self,root):
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
        if type(self.receiver) == str:
            letterboxes[self.receiver] = [self]
        elif not self.content in [m.content for m in letterboxes[self.receiver]]:
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
    order_out = instance.fg.topo_order_out(root)
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
                if neigh not in [m.sender for m in letterboxes[node]]:
                    message.send(letterboxes)
    

    return letterboxes