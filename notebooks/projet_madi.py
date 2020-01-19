import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import math
import numpy as np
from factorGraph import FactorGraph
from treeSumProduct import TreeSumProductInference
from treeMaxProduct import TreeMaxProductInference
from treeMaxSum import TreeMaxSumInference
from itertools import product


def fill_binary_potential(potential):
    p = np.array(potential.toarray())
    shpe = p.shape
    p = np.reshape(p,p.size)
    items = list(product(*[[0,1]]*len(shpe)))
    for i,e in enumerate(p):
        p[i] = 1 if sum(items[i]) % 2 == 0 else 0
    p = np.reshape(p,shpe)
    if shpe != (1,):
        for item in items:
            potential[item] = p[item]
    else:
        potential.fillWith(0)

def buildLDPC(bits,parity):
    bn=gum.BayesNet('LDPC')
    for bit in bits:
        bn.add(gum.LabelizedVariable(bit,bit+'name',2))
    for pc, bits_ in parity.items():
        bn.add(gum.LabelizedVariable(pc,pc+'name',2))
        for bit in bits_:
            bn.addArc(bit,pc)
            
    for i in bn.nodes():
        p = bn.cpt(i)
        if len(p.toarray().shape) == 1:
            p[0] = 0.5
            p[1] = 0.5
            p = p.normalizeAsCPT()
        else:
            fill_binary_potential(bn.cpt(i))
    
    return bn