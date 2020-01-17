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
                    


class TreeSumProductInference:
    def __init__(self,f):
        self.fg = f
        self.tables = {}

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        
        # Les noeuds à visiter, initialisés au feuilles de l'arbre
        to_visit = self.fg.leaves()
        # Les noeuds pour lequel les probalilités sont connues
        visited = []
        # La boîte aux lettres de chaque noeuds sous la forme d'un dictionnaire de listes de potentiels
        letterbox = {v.name() : [] for v in self.fg.variables}
        
        while len(to_visit) > 0:
            next_generation = []
            # Parcours des noeuds de la génération en cours
            for node in to_visit:
                # Si un noeud a suffisament de messages dans sa boîte aux lettres pour calculer ses probabilités
                if len(letterbox[node]) >= len(self.fg.edges[node].var_names) - 1:
                    visited.append(node)
                    # Calcul du potentiel du noeud
                    potential = self.fg.edges[node]
                    for m in letterbox[node]:
                        potential = potential * m
                    potential = potential.margSumIn(node)
                    
                    # Mise à jour de la table des probabilités de chaque noeuds
                    self.tables[node] = potential
                    
                    # Parcours des voisins non explorés (ici ce seront les enfants du noeud)
                    neighbours = self.fg.neighbours(node)
                    for n in neighbours:
                        if not n in visited:
                            # Envoi du message du potentiel à l'enfant
                            letterbox[n].append(potential)
                            # Ajout de l'enfant dans les noeuds à explorer
                            if not n in next_generation:
                                next_generation.append(n)
                else:
                    # Si le noeud n'a pas suffisament de message, il faudra à nouveau l'exporer
                    next_generation.append(node)
            
            # Les noeuds à explorer sont actualisés comme ceux de la génération suivante 
            to_visit = next_generation


    def posterior(self, variable):
        """ retourne la distribution de la variable sous la forme d'un `gum.Potential` """
        try:
            return self.tables[variable]
        except KeyError:
            print("{} not found in variables, try to makeInference on the object.".format(variable))
            
class TreeMaxProductInference:
    def __init__(self,f):
        self.fg = f
        self.most_likely_values = {}

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        
        # Les noeuds à visiter, initialisés au feuilles de l'arbre
        to_visit = self.fg.leaves()
        # Les noeuds pour lequel les probalilités sont connues
        visited = []
        # La boîte aux lettres de chaque noeuds sous la forme d'un dictionnaire de listes de potentiels
        letterbox = {v.name() : [] for v in self.fg.variables}
        
        while len(to_visit) > 0:
            next_generation = []
            # Parcours des noeuds de la génération en cours
            for node in to_visit:
                # Si un noeud a suffisament de messages dans sa boîte aux lettres pour calculer ses probabilités
                if len(letterbox[node]) >= len(self.fg.edges[node].var_names) - 1:
                    visited.append(node)
                    # Calcul du potentiel du noeud
                    potential = self.fg.edges[node]
                    for m in letterbox[node]:
                        potential = potential * m
                    potential = potential.margMaxIn(node)
                    best_index = potential.argmax()[0][node]
                    potential[best_index] = 1                    
                    potential[1-best_index] = 0
                    
                    # Mise à jour de la table des probabilités de chaque noeuds
                    self.most_likely_values[node] = best_index
                    
                    # Parcours des voisins non explorés (ici ce seront les enfants du noeud)
                    neighbours = self.fg.neighbours(node)
                    for n in neighbours:
                        if not n in visited:
                            # Envoi du message du potentiel à l'enfant
                            letterbox[n].append(potential)
                            # Ajout de l'enfant dans les noeuds à explorer
                            if not n in next_generation:
                                next_generation.append(n)
                else:
                    # Si le noeud n'a pas suffisament de message, il faudra à nouveau l'exporer
                    next_generation.append(node)
            
            # Les noeuds à explorer sont actualisés comme ceux de la génération suivante 
            to_visit = next_generation


    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return self.most_likely_values

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

class TreeMaxSumInference:
    def __init__(self,f):
        self.fg = f
        self.most_likely_values = {}

    def makeInference(self):
        """ effectue les calculs de tous les messages """
        
        # Les noeuds à visiter, initialisés au feuilles de l'arbre
        to_visit = self.fg.leaves()
        # Les noeuds pour lequel les probalilités sont connues
        visited = []
        # La boîte aux lettres de chaque noeuds sous la forme d'un dictionnaire de listes de potentiels
        letterbox = {v.name() : [] for v in self.fg.variables}
        
        while len(to_visit) > 0:
            next_generation = []
            # Parcours des noeuds de la génération en cours
            for node in to_visit:
                # Si un noeud a suffisament de messages dans sa boîte aux lettres pour calculer ses probabilités
                if len(letterbox[node]) >= len(self.fg.edges[node].var_names) - 1:
                    visited.append(node)
                    # Calcul du potentiel du noeud
                    potential = self.fg.edges[node]
                    potential = f_transform(potential,np.log)
                    for m in letterbox[node]:
                        m_transform = f_transform(m,np.log)
                        potential = potential + m_transform
                    potential = f_transform(potential,np.exp)
                    potential = potential.margMaxIn(node)
                    best_index = np.argmax([potential[0], potential[1]])
                    potential[best_index] = 1                    
                    potential[1-best_index] = 0
                    
                    
                    
                    
                    # Mise à jour de la table des probabilités de chaque noeuds
                    self.most_likely_values[node] = best_index
                    
                    # Parcours des voisins non explorés (ici ce seront les enfants du noeud)
                    neighbours = self.fg.neighbours(node)
                    for n in neighbours:
                        if not n in visited:
                            # Envoi du message du potentiel à l'enfant
                            letterbox[n].append(potential)
                            # Ajout de l'enfant dans les noeuds à explorer
                            if not n in next_generation:
                                next_generation.append(n)
                else:
                    # Si le noeud n'a pas suffisament de message, il faudra à nouveau l'exporer
                    next_generation.append(node)
            
            # Les noeuds à explorer sont actualisés comme ceux de la génération suivante 
            to_visit = next_generation


    def argmax(self):
        """ retourne un dictionnaire des valeurs des variables pour le MAP """
        return self.most_likely_values