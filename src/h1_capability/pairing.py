"""Pair all candidates before seeing reachability outcomes; retain all rejections."""
import itertools
import numpy as np

def match_pairs(force_records,tolerance=0.1,floor_n=1.):
    out=[]
    for a,b in itertools.combinations(force_records,2):
        reasons=[]
        if a['status']!='ACCEPTED' or b['status']!='ACCEPTED':reasons.append('candidate_filter')
        aa=np.array(a.get('force_match_vector_n',[]));bb=np.array(b.get('force_match_vector_n',[]))
        gap=None
        if aa.shape==(12,) and bb.shape==(12,):
            gap=float(np.max(abs(aa-bb)/np.maximum(floor_n,(abs(aa)+abs(bb))/2)))
            if gap>tolerance:reasons.append('force_vector_mismatch')
        else:reasons.append('missing_force_capacity')
        out.append(dict(a=a['id'],b=b['id'],status='MATCHED' if not reasons else 'REJECTED',
                        reasons=reasons,max_relative_gap=gap))
    return out
