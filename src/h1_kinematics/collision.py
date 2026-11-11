"""FCL supplied triangle-mesh surface checks with conservative hull broad phase.

A separated hull certifies separated surfaces. For close/overlapping hulls,
use the original triangles; this avoids wrist hull false positives. Open meshes
cannot certify closed solid occupancy or detect every containment case. Same rigid assembly and directly connected assemblies excluded;
no learned/zero-pose exemptions. No floor/person/bed geometry is supplied.
"""
import itertools
import numpy as np
import fcl

class CollisionChecker:
    def __init__(self,asset):
        self.asset=asset
        self.meshes={n:asset.mesh(l.find('collision')) for n,l in asset.links.items()
                     if l.find('collision') is not None}
        self.hulls={n:m.convex_hull for n,m in self.meshes.items()}
        self.objects={}; self.exact={}
        for n,h in self.hulls.items():
            faces=np.column_stack([np.full(len(h.faces),3),h.faces]).astype(np.int32).ravel()
            self.objects[n]=fcl.CollisionObject(fcl.Convex(h.vertices,len(h.faces),faces))
            m=self.meshes[n]; bvh=fcl.BVHModel(); bvh.beginModel(len(m.vertices),len(m.faces))
            bvh.addSubModel(m.vertices,m.faces); bvh.endModel()
            self.exact[n]=fcl.CollisionObject(bvh)
        group={n:n for n in asset.links}
        def find(n):
            while group[n]!=n: n=group[n]
            return n
        for j in asset.joints.values():
            if j.get('type')=='fixed':
                group[find(j.find('child').get('link'))]=find(j.find('parent').get('link'))
        connected=set()
        for j in asset.joints.values():
            a=find(j.find('parent').get('link')); b=find(j.find('child').get('link'))
            connected.add(frozenset([a,b]))
        self.pairs=[]; self.excluded=[]
        for a,b in itertools.combinations(self.objects,2):
            ga,gb=find(a),find(b)
            reason='same_fixed_assembly' if ga==gb else (
                'directly_connected_assemblies' if frozenset([ga,gb]) in connected else None)
            if reason: self.excluded.append(dict(a=a,b=b,reason=reason))
            else: self.pairs.append((a,b))

    def check(self,frames,clearance_m=0.002):
        for n,o in self.objects.items():
            t=frames[n]; placement=fcl.Transform(t[:3,:3],t[:3,3])
            o.setTransform(placement); self.exact[n].setTransform(placement)
        collisions=[]; close=[]; narrow_pairs=[]; minimum=float('inf'); closest=None
        for a,b in self.pairs:
            oa,ob=self.objects[a],self.objects[b]
            result=fcl.CollisionResult()
            hit=fcl.collide(oa,ob,fcl.CollisionRequest(),result)>0
            distance=-1. if hit else float(fcl.distance(oa,ob,fcl.DistanceRequest(),fcl.DistanceResult()))
            if distance<clearance_m:
                narrow_pairs.append([a,b])
                oa,ob=self.exact[a],self.exact[b]
                hit=fcl.collide(oa,ob,fcl.CollisionRequest(),fcl.CollisionResult())>0
                distance=-1. if hit else float(fcl.distance(oa,ob,fcl.DistanceRequest(),fcl.DistanceResult()))
            if not np.isfinite(distance): raise FloatingPointError('Nonfinite FCL distance')
            if distance<minimum: minimum,closest=distance,[a,b]
            if hit: collisions.append([a,b])
            elif distance<clearance_m: close.append(dict(pair=[a,b],distance_m=distance))
        return dict(passed=not collisions and not close,collision_pairs=collisions,
                    below_clearance=close,min_distance_m=minimum,closest_pair=closest,
                    checked_pairs=len(self.pairs),clearance_m=clearance_m,narrow_phase_pairs=narrow_pairs,
                    criterion='supplied_mesh_surface_clearance; no solid-occupancy guarantee')
