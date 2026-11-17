"""Boundary-loop identification and geometric sleeve progress proxies (not forces)."""
import numpy as np

def opening(points,normal=None):
    """Best-fit cuff plane and ordered polygon; metres, no force interpretation."""
    p=np.asarray(points,dtype=float);center=p.mean(axis=0)
    _,_,vh=np.linalg.svd(p-center,full_matrices=False)
    n=vh[-1]
    if normal is not None and n@normal<0:n=-n
    u=vh[0];v=np.cross(n,u);xy=np.c_[(p-center)@u,(p-center)@v]
    area=.5*abs(float(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))))
    spans=np.ptp(xy,axis=0)
    return dict(center=center,normal=n,u=u,v=v,polygon=xy,area_m2=area,spans_m=spans,planarity_rms_m=float(np.std((p-center)@n)))

def contains_section(points,center,radius,axis,clearance=.003):
    """Require a finite-radius section disk inside the actual projected cuff polygon."""
    frame=opening(points,axis);p=frame['polygon'];axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    alignment=float(frame['normal']@axis)
    # Intersect the arm centreline with the cuff plane. A tilted cylinder cuts
    # an ellipse; its circumscribed circle gives a conservative containment test.
    center=np.asarray(center,dtype=float)
    intersection=center+axis*float((frame['center']-center)@frame['normal'])/max(alignment,1e-8)
    required=radius/max(alignment,1e-8)+clearance
    c=np.array([(intersection-frame['center'])@frame['u'],(intersection-frame['center'])@frame['v']])
    x,y=c;inside=False
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    a=p;b=np.roll(p,-1,axis=0);ab=b-a;t=np.clip(np.sum((c-a)*ab,axis=1)/np.maximum(np.sum(ab*ab,axis=1),1e-14),0,1)
    gap=float(np.min(np.linalg.norm(c-(a+t[:,None]*ab),axis=1)))
    return dict(inside=bool(inside and gap>=required and alignment>=.90 and frame['planarity_rms_m']<.012),center_inside=bool(inside),edge_clearance_m=gap,required_m=float(required),normal_alignment=alignment,plane_distance_m=float((center-frame['center'])@frame['normal']),area_m2=frame['area_m2'],planarity_rms_m=frame['planarity_rms_m'])

def opened_preset(points,cuff,center,axis,inflate_length=.18):
    """One-time deformed initial state, original FEM rest mesh/material unchanged.

    Wrap the flattened sleeve's two faces into semicircles. Bend the remote shirt
    body downward to keep it outside the robot's torso at initialization.
    This is a simulation preset, not an executed tabletop grasp.
    """
    p=np.asarray(points);c=p[cuff].mean(axis=0);axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    side=np.array([0.,1.,0.]);up=np.cross(axis,side);up/=np.linalg.norm(up)
    perimeter=float(np.linalg.norm(np.roll(p[cuff],-1,axis=0)-p[cuff],axis=1).sum());radius=perimeter/(2*np.pi)
    zmid=.5*(p[cuff,2].max()+p[cuff,2].min())
    a=np.maximum(p[:,0]-c[0],0);y=p[:,1]-c[1];z=p[:,2]-zmid
    weight=np.clip(1-a/inflate_length,0,1)
    weight*=np.exp(-np.maximum(np.abs(y)-np.max(np.abs(p[cuff,1]-c[1])),0)/.035)
    # Arc length preserves the tiny seam edges too. A sign(z) mapping would
    # stretch some of those edges >10x when the two flattened faces separate.
    boundary=p[cuff,1:3];seg=np.roll(boundary,-1,axis=0)-boundary
    lengths=np.linalg.norm(seg,axis=1);arc=np.r_[0.,np.cumsum(lengths)[:-1]]
    phase_offset=2*np.pi*arc[np.argmax(boundary[:,0])]/perimeter
    boundary_phase=2*np.pi*arc/perimeter-phase_offset
    sign=1 if np.sum(np.sin(boundary_phase)*(boundary[:,1]-zmid))>=0 else -1
    distance2=np.full(len(p),np.inf);phase=np.zeros(len(p))
    for i,(v,edge,length) in enumerate(zip(boundary,seg,lengths)):
        t=np.clip((p[:,1:3]-v)@edge/max(length*length,1e-16),0,1)
        dist=np.sum((p[:,1:3]-v-t[:,None]*edge)**2,axis=1);mask=dist<distance2
        phase[mask]=sign*(2*np.pi*(arc[i]+t[mask]*length)/perimeter-phase_offset);distance2[mask]=dist[mask]
    yy=(1-weight)*y+weight*radius*np.cos(phase)
    zz=(1-weight)*z+weight*radius*np.sin(phase)
    # Isometric centreline: straight sleeve, quarter-turn, then hanging body.
    straight=.10;bend=.06;theta=np.clip((a-straight)/bend,0,np.pi/2)
    back=np.minimum(a,straight)+bend*np.sin(theta)
    down=bend*(1-np.cos(theta))+np.maximum(a-straight-bend*np.pi/2,0)
    out=np.asarray(center)-back[:,None]*axis-down[:,None]*up+yy[:,None]*side+zz[:,None]*up
    return out,dict(radius_m=radius,perimeter_m=perimeter,initialization='analytic one-time opened sleeve preset; FEM rest state unchanged')

def grasp_patches(points,cuff,count=5,triangles=None,depth=.025,half_arc=.012):
    """Connected surface strips at opposite rim sides, with physical dimensions.

    Tiny seam edges make a fixed vertex count a poor physical patch size. Grow
    only connected triangles inside a 24 x 25 mm strip, never a second wall.
    """
    n=len(cuff);result={}
    points=np.asarray(points);frame=opening(points[cuff]);normal=frame['normal']
    # On this asset the shirt lies on the negative-normal side of the cuff.
    if normal[0]<0:normal=-normal
    adjacency={}
    if triangles is not None:
        for tri in triangles:
            for a in tri:adjacency.setdefault(int(a),set()).update(map(int,tri))
    for side,sign in [('L',1),('R',-1)]:
        i=int(np.argmax(sign*points[cuff,1]))
        if triangles is None:result[side]=cuff[(i+np.arange(count)-count//2)%n];continue
        center=points[cuff[i]];radial=center-frame['center'];radial/=np.linalg.norm(radial);tangent=np.cross(normal,radial)
        delta=points-center
        mask=(delta@normal>=-depth)&(delta@normal<=.001)&(np.abs(delta@tangent)<=half_arc)&(np.abs(delta@radial)<=.008)
        selected={int(cuff[i])};queue=list(selected)
        while queue:
            for nxt in adjacency[queue.pop()]:
                if mask[nxt] and nxt not in selected:selected.add(nxt);queue.append(nxt)
        result[side]=np.array(sorted(selected),dtype=np.int64)
    if set(result['L'])&set(result['R']):raise ValueError('Grasp patches overlap')
    return result

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
