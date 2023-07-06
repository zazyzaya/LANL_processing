import glob 

from joblib import Parallel, delayed
import torch 
from torch_geometric.data import Data
from tqdm import tqdm 

from globals import *

WORKERS = 32

def load_one(idx,fname,write_to):
    f = open(fname, 'r')
    stem = fname.split('/')[-1].split('.')[0]
    
    x = dict() 
    src,dst = [],[]
    
    one_hots = []
    quants = []
    
    ts = []
    ys = []

    line = f.readline()
    
    prog = tqdm(desc=str(idx))
    while line:
        t,edge,ntypes,oh,q,y = eval(f'[{line}]')
        
        s,d = edge; s_x,d_x = ntypes 
        x[s] = s_x; x[d] = d_x 
        src.append(s); dst.append(d)

        ts.append(t)
        quants.append(q)
        ys.append(y)

        one_hots.append([o for o in oh if o is not None])
        
        line = f.readline()
        prog.update()
        
    num_e = len(src)
    prog.close()
    
    oh = torch.zeros(num_e, NUM_ETYPES) 
    for i in range(num_e):
        oh[i, one_hots[i]] = 1. 
    
    # The duration, npkts, nbytes can vary so wildly
    # Duration isn't as bad (0 - ~70), but the other two 
    # range from 0 to 1e6, and 1e8 respectively 
    quants = torch.log(torch.tensor(quants))
    
    ei = torch.tensor([src,dst])
    ys = None if sum(ys) == 0 else torch.tensor(ys)
    ts = torch.tensor(ts)
    ew = torch.cat([quants, oh], dim=1)

    torch.save(
        Data(edge_index=ei, edge_attr=ew, y=ys, ts=ts),
        write_to+stem+'.pt'
    )

    return x 

def load_all(write_to='torch_files/'):
    files = glob.glob(READ_FROM+'*.txt')
    xs = Parallel(n_jobs=WORKERS, prefer='processes')(
        delayed(load_one)(i,f,write_to) for i,f in enumerate(files)
    )

    x_dict = xs.pop(-1)
    [x_dict.update(xs[i]) for i in range(len(xs))]
    
    x = torch.zeros(NUM_NODES, NUM_NODE_TYPES)
    for i in range(x.size(0)):
        if (idx := x_dict.get(i)) is not None:
            x[i, idx] = 1. 

    torch.save(x, write_to+'features.pt')

if __name__ == '__main__':
    load_all()