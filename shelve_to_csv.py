import shelve 

def convert(fname):
    db = shelve.open(f'output/{fname}.db', 'r')
    with open(f'output/{fname}.csv', 'w+') as f:
        for k,v in db.items(): 
            f.write(f'{k},{v}\n')

    print(fname,len(db))

convert('eid_map')
convert('nid_map')