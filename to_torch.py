import torch 
from tqdm import tqdm 

# Three quantitative feaatures
NUM_QUANT = 3 
NUM_OH = None # TODO 

def load_one(fname):
    f = open(fname, 'r')
    
    x = dict() 
    src,dst = [],[]
    
    one_hots = []
    max_oh = 0
    quants = []
    
    ts = []
    ys = []

    line = f.readline()
    
    prog = tqdm(desc='lines read')
    while line:
        t,edge,ntypes,oh,q,y = eval(f'[{line}]')
        
        s,d = edge; s_x,d_x = ntypes 
        x[s] = s_x; x[d] = d_x 
        src.append(s); dst.append(d)

        ts.append(t)
        quants.append(q)
        ys.append(y)

        one_hots.append(oh)
        max_oh = max(max_oh, max(oh))
        
    num_e = len(src)
    
    oh = torch.zeros(num_e, max_oh+1) 
    for i in range(num_e):
        oh[i, one_hots[i]] = 1. 
    
    quants = torch.tensor(quants)
    ei = torch.tensor([src,dst])
    ys = None if sum(ys) == 0 else torch.tensor(ys)
    
