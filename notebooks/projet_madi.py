import pydotplus as dot
from IPython.display import SVG


def factor_repr(f):
    string = "p"+f.var_names[-1]
    if len(f.var_names) > 1:
        string += "g"
        for vn in f.var_names[:-1]:
            string += vn
    return string

class FactorGraph:
    """
    variables : Une liste de gum.DiscreteVariable
    factors : Une liste de gum.Potential
    edges : Une liste de couples (gum.DiscreteVariable, gum.Potential)
    """
    
    def addVariable(self,v):
        """ Ajout de variable sous la forme de gum.DiscreteVariable """
        self.variables.append(v)
    
    def addFactor(self,p):
        """ Ajout de facteurs sous la forme de gum.Potential """
        self.factors.append(p)
    
    def build(self,bn):
        """ Construit un factor graph à partir d'un gum.BayesNet """
        self.variables = []
        self.factors = []
        self.edges = {}
        for i in bn.nodes():
            self.addVariable(bn.variable(i))
            self.addFactor(bn.cpt(i))
            self.edges[bn.variable(i).name()] = bn.cpt(i)
                

    def show(self):
        """ Affichage d'un factor graph """
        string = """
        graph FG {
            layout=neato;
            node [shape=rectangle,margin=0.04,
                  width=0,height=0, style=filled,color="coral"];
            """
        for v in self.variables:
            string += v.name() + ";"
        string += """
            node [shape=point];
            """
        for f in self.factors:
            string += factor_repr(f) + ";"
            
        string += """
            edge;
            """
        
        for _,f in self.edges.items():
            for vn in f.var_names:
                string += factor_repr(f) + "--" + vn + ";\n"

        string += "}"
        g=dot.graph_from_dot_data(string)
        return SVG(g.create_svg())
    
    def neighbours(self,variable):
        n = []
        for v,f in self.edges.items():
            if variable in f.var_names:
                for vn in f.var_names:
                    if not vn in n and vn != variable and (vn == v or v == variable):
                        n.append(vn)
        return n
    
    def leaves(self):
        l = []
        for v in self.variables:
            if self.edges[v.name()].toarray().size == 2:
                l.append(v.name())
        return l


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
                    print("=====")
                    print(node)
                    print(potential)
                    print(letterbox[node])
                    for m in letterbox[node]:
                        potential = potential * m
                    print(potential)
                    potential = potential.margMaxIn(node)
                    print(potential)
                    
                    
                    # Mise à jour de la table des probabilités de chaque noeuds
                    self.most_likely_values[node] = potential
                    
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
        return {v:p.argmax() for v,p in self.most_likely_values.items()}

