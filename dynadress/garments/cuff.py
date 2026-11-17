"""Boundary-loop identification and geometric sleeve progress proxies (not forces)."""
import numpy as np

def boundary_loops(triangles):
    edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    edges,counts=np.unique(edges,axis=0,return_counts=True);adj={}
    for a,b in edges[counts==1]:adj.setdefault(int(a),[]).append(int(b));adj.setdefault(int(b),[]).append(int(a))
    if any(len(n)!=2 for n in adj.values()):raise ValueError('Nonmanifold garment boundary')
    loops=[];seen=set()
    for start in adj:
        if start in seen:continue
        loop=[];prev=None;v=start
        while v not in seen:
            seen.add(v);loop.append(v);n=adj[v];prev,v=v,(n[0] if n[0]!=prev else n[1])
        loops.append(np.asarray(loop,dtype=np.int64))
    return loops

def progress(cuff_points,hand_tip,elbow,target=.12):
    center=np.mean(cuff_points,axis=0);axis=np.asarray(elbow)-hand_tip;length=np.linalg.norm(axis)
    if length<1e-6:raise ValueError('Degenerate human segment')
    distance=float(np.dot(center-hand_tip,axis/length))
    return {'axial_progress_m':distance,'target_m':target,'proxy_only':True,'completion':False,
            'completion_note':'Axial projection alone does not prove hand containment or sleeve insertion'}
