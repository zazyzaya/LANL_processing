import gzip 

import torch 
from tqdm import tqdm 

DATA_DIR = '/mnt/raid1_ssd_4tb/datasets/LANL15/'
TORCH_DIR = DATA_DIR + 'torch_data/'
REDLOG = DATA_DIR + 'redteam.txt.gz'
OUT_DIR = 'tmp/'

'''
I messed something up in the combine.py file s.t. nothing has 
labels. This seemed faster than rebuilding everything all
over again. Only takes about 5 mins 
'''

def get_node_map():
    nmap = dict()
    with open(TORCH_DIR+'nid_map.csv', 'r') as f:
        line = f.readline()
        
        while line:
            node_str,nid = line.split(',')
            nmap[node_str] = int(nid)
            line = f.readline()

    return nmap 

def label_g(g, ts, src, dst):
    relevant = g.ts == ts
    
    ei = g.edge_index
    red = relevant.logical_and(ei[0] == src).logical_and(ei[1] == dst)
    return red.long()

def main():
    red_f = gzip.open(REDLOG, 'rt')
    red = red_f.readline() 
    nmap = get_node_map()
    
    last_file = this_file = None 
    g = None 
    prog = tqdm(total=1392)
    while red: 
        # Format: 
        # ts,    u@dom,    src_c, dst_c
        # 151648,U748@DOM1,C17693,C728
        ts,u_at_dom,_,dst_c = red.strip().split(',')
        src = u_at_dom.split('@')[0]
        dst = dst_c 
        ts = int(ts)

        this_file = ts // 3600
        
        # Only runs on first iter of the loop
        if g is None: 
            g = torch.load(TORCH_DIR + f'{this_file}.pt')
            last_file = this_file 
            prog.n = this_file-1
            prog.update()

        # Save changes to last file if anomaly occurs in 
        # a new section of the graph
        if this_file != last_file:
            torch.save(g, OUT_DIR+f'{last_file}.pt')
            g = torch.load(TORCH_DIR + f'{this_file}.pt')
            last_file = this_file
            prog.n = this_file -1 
            prog.update() 

        y = label_g(g, ts, nmap[src], nmap[dst])

        # In case >1 anom in an hour
        if g.y is not None:
            g.y += y
        else: 
            g.y = y 
    
        red = red_f.readline() 
    
    # One last save before we're done 
    prog.n = 1392-1
    prog.update()
    g = torch.load(TORCH_DIR + f'{this_file}.pt')
    prog.close()

if __name__ == '__main__':
    main()