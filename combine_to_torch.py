import asyncio
import gzip
import multiprocessing as mp 
from threading import Lock, Thread

import torch 
from torch_geometric.data import Data
from tqdm import tqdm 

from globals import *

READ = '/home/ead/datasets/LANL15/'
WRITE = 'output/'
FILE_NAMES = [READ+s+'.txt.gz' for s in ['auth', 'dns', 'flows']]#, 'proc']]
IO_HANDLES = {i:gzip.open(f, 'rt') for i,f in enumerate(FILE_NAMES)}
LABELS = gzip.open(READ+'redteam.txt.gz', 'rt')

MAX_CACHE = 2**16

with open('maps/nid_map.csv', 'r') as f:
    NID_MAP = dict() 
    line = f.readline()

    while line: 
        s,i = line.split(',')
        NID_MAP[s] = int(i.strip())
        line = f.readline()

with open('maps/eid_map.csv', 'r') as f:
    EDGE_MAP = dict() 
    line = f.readline()

    while line: 
        s,i = line.split(',')
        EDGE_MAP[s] = int(i.strip())
        line = f.readline()

SPLIT = 3600 # Arbitrarally set to 1hrs per file
FLUSH_AFTER = float('inf') # Doesn't seem to be a problem on this machine
MAX_SUBPROCS = 32

'''
Example of redlog and corresponding auth log
Red:  '151036,U748@DOM1,C17693,C305'
Auth: '151036,U748@DOM1,U748@DOM1,C17693,C305,NTLM,Network,LogOn,Success'

Maybe only src user and dst computer matter? 
Seems like src and dst user are very often the same? 
'''

def get_nid(node_str):
    return NID_MAP.get(node_str)

def get_edge_type(edge_str):
    return EDGE_MAP.get(edge_str)

def get_ntype(node_str):
    if node_str.startswith('U'):
        return 0
    elif node_str.startswith('C'):
        return 1
    elif node_str.startswith('A'): # Anonymous Login
        return 2
    else:
        return 3

def parse_auth(line, next_mal):
    '''
    Every line gives us src_u, dst_u, src_c, dst_c meaning 
    src_user uses src_computer to authenticate with dst_user at dst_computer

    We will simply add an edge from the src_u to the dst_c as this seems 
    to be what matters in the red log(?)

    Edge feats: ONE HOT
    '''
    ts, src_u,dst_u, src_c,dst_c, proto,a_type,orientation,success = line.split(',')
    
    src_x = get_ntype(src_u)
    dst_x = get_ntype(dst_c)

    if src_x > 1 or dst_x > 1:
        return int(ts), None, None, None, None, None

    snid = get_nid(src_u)
    dnid = get_nid(dst_c)
    if snid == None or dnid == None:
        return int(ts), None, None, None, None, None

    is_mal = False 
    if next_mal and next_mal[0] == ts:
        if src_u == next_mal[1] and src_c == next_mal[2] and dst_c == next_mal[3]:
            is_mal=True

    return \
        int(ts), \
        [
            snid, 
            dnid
        ], \
        [
            src_x,
            dst_x
        ], \
        [
            get_edge_type(proto), 
            get_edge_type(a_type), 
            get_edge_type(orientation)
        ], [0,0,0], is_mal 

def parse_dns(line, *args):
    '''
    Edge feats: ONE HOT 
    '''
    ts,src_c,dst_c = line.split(',')

    src_x = get_ntype(src_c)
    dst_x = get_ntype(dst_c)
    if src_x > 1 or dst_x > 1:
        return int(ts), None, None, None, None, None

    snid = get_nid(src_c)
    dnid = get_nid(dst_c)
    if snid == None or dnid == None:
        return int(ts), None, None, None, None, None
    
    return int(ts), \
        [snid, dnid], \
        [src_x, dst_x], \
        [get_edge_type('DNS')], \
        [0,0,0], False

def parse_flows(line, *args):
    '''
    time,duration,source computer,source port,destination computer,destination port,protocol,packet count,byte count

    Ignoring ports, but duration seems like a useful (non-onehot) value

    Edge feats: ONE HOT and QUANTITATIVE
    '''
    ts,duration,src_c,src_p,dst_c,dst_p,proto,n_pkts,n_bytes = line.split(',')
    
    src_x = get_ntype(src_c)
    dst_x = get_ntype(dst_c)
    if src_x > 1 or dst_x > 1:
        return int(ts), None, None, None, None, None

    snid = get_nid(src_c)
    dnid = get_nid(dst_c)
    if snid == None or dnid == None:
        return int(ts), None, None, None, None, None
    
    return int(ts), \
        [snid, dnid], \
        [src_x, dst_x], \
        [get_edge_type(proto)], \
        [int(duration), int(n_pkts), int(n_bytes)], \
        False 

def parse_proc(line, *args):
    '''
    time, source user, dst computer, process, start/end
    
    Seems like the only time it isn't a self loop of C1$@DOM1, C1 is if it's a user
    so we will only add edges of the form U[\d+], C[\d+]

    Edge feats: ONE HOT
    '''
    ts,src,dst,p,st_en = line.split(',')

    if not src.startswith('U'):
        return int(ts), None, None, None, None, None
    
    src_x = get_ntype(src)
    dst_x = get_ntype(dst)
    if src_x > 1 or dst_x > 1:
        return int(ts), None, None, None, None, None

    snid = get_nid(src)
    dnid = get_nid(dst)
    if snid == None or dnid == None:
        return int(ts), None, None, None, None, None
    
    return int(ts), \
        [snid, dnid], \
        [src_x, dst_x], \
        [get_edge_type(p), get_edge_type(st_en)], \
        [], False 

def parse_all():
    cur_time = st_time = 0 
    cur_file = f'{WRITE}/{cur_time}.pt'
    buffer = []

    parsers = {
        0: parse_auth, 1: parse_dns, 
        2: parse_flows, 3: parse_proc
    }

    lines = {
        k:IO_HANDLES[k].readline().strip() 
        for k in IO_HANDLES.keys()
    }

    def parse_one_second(idx, line, cur_time, cur_red):
        ret_val = []
        ts, edge, ntypes, oh, val, label = parsers[idx](line, cur_red)
        
        while ts == cur_time:
            # All values other than ts will be None if missing data
            if edge: 
                ret_val.append(
                    (ts, edge, ntypes, oh, val, label)
                )
            line = IO_HANDLES[idx].readline().strip()
            
            # Check there is still data to read
            if line:
                ts, edge, ntypes, oh, val, label = parsers[idx](line, cur_red)
            else:
                break 

        return ret_val, line, idx

    def flush(f, edges):
        '''
        Edges formatted as 
        ts,(src,dst),(src_x, dst_x),(one hot idxs),(quantitative values),y
        '''
        # Checking we're not touching mem that's allocated in the main thread
        #print(id(edges), 'in thread') 
        #out_str = ''
        src,dst = [],[]
        edge_feats = []
        ts = []
        ys = []

        e = edges.pop(0) 
        while edges:
            t,(s,d),_,oh,quant,y = e 
            
            oh = [val for val in oh if val is not None]
            ef = torch.zeros(NUM_ETYPES + NUM_QUANT)
            ef[oh] = 1. 
            ef[-3:] = torch.tensor(quant)
            edge_feats.append(ef)

            src.append(s)
            dst.append(d)
            ts.append(t)
            ys.append(y)

            e = edges.pop(0)

        ys = torch.tensor(ys)
        if ys.sum() == 0:
            ys = None 

        torch.save(
            Data(
                edge_index = torch.tensor([src,dst]),
                edge_attr = torch.stack(edge_feats),
                ts = torch.tensor(ts),
                ys = ys 
            ), f
        )

        #f.write(out_str)
        print(f"Finished writing {f.split('/')[-1]}")


    cur_red = LABELS.readline().strip()
    cur_red = cur_red.split(',')
    red_time = int(cur_red[0])

    # Approx end from zcat dns.txt.gz | tail 
    prog = tqdm(desc='Seconds parsed', total=NUM_SECONDS)
    file_num = 0
    writers = []
    while cur_time <= NUM_SECONDS: 
        response = [ 
            parse_one_second(idx, lines[idx], cur_time, cur_red)
            for idx in IO_HANDLES.keys()
        ]

        edges, last_lines, idxs = zip(*response)
        for i in range(len(idxs)):
            # Add edge data 
            buffer += edges[i]
            
            # Close any files that finished
            # This does not appear to work, so after 
            # it finished running, I just took the last time stamp
            # and made the loop check for that. leaving this here
            # just in case it breaks something to take it out
            if not last_lines[i]:
                IO_HANDLES[idxs[i]].close()
                del IO_HANDLES[idxs[i]]

            # Update last lines
            else:
                lines[idxs[i]] = last_lines[i]
            
        # Advance to next redteam event if the last one was captured
        if red_time == cur_time:
            cur_red = LABELS.readline().strip()
            if cur_red:
                cur_red = cur_red.split(',')
                red_time = int(cur_red[0])
            else: 
                cur_red = [None] * 4 
                red_time = -1 

        '''
        # If the buffer is full, write out edges
        if len(buffer) >= FLUSH_AFTER:
            flush(cur_file, buffer)
        '''

        # Go to next second
        cur_time += 1
        prog.update()

        # If the timestamp is complete, write out edges, and make new file 
        if cur_time == (st_time + SPLIT): 
            # Multithreading seems to just slow stuff down 
            # since the threads share the same IO buffer
            if len(writers) > MAX_SUBPROCS: 
                writers.pop(0).join()

            p = mp.Process(
                group=None, target=flush, 
                args=(cur_file, buffer)
            )
            p.start()
            writers.append(p)

            file_num += 1
            buffer = []
            
            # Checking that buffer is a new memory address and 
            # we aren't making race conditions in the worker thread
            #print(id(buffer), 'out of thread')
            
            st_time = cur_time
            cur_file = f'{WRITE}/{file_num}.pt'

    prog.close()
    [w.join() for w in writers]

    # One last flush to finish it off, then we're done 
    flush(cur_file, buffer)
    NID_MAP.close()
    EDGE_MAP.close()

if __name__ == '__main__': 
    parse_all()