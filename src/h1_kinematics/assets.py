"""Read-only source audit and explicit mesh URI resolution. SI per URDF convention."""
from pathlib import Path
import hashlib
import json
import zipfile
import tempfile
import os
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ARCHIVE_SHA256 = '5e40a5558cf3ed0dd59d1e9a08fa38944ddbb7c48e1612d0b7010a39cbeeb977'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def unpack(archive, destination):
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Archive checksum differs from audited source')
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            p = Path(i.filename)
            if p.is_absolute() or '..' in p.parts or (i.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Unsafe ZIP entry: {i.filename}')
        # Existing assets are immutable; independent runs may share this directory.
        # Publish each new file atomically so a reader cannot see a partial STL.
        for i in z.infolist():
            target = destination / i.filename
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Extraction target escapes through a symlink')
            if i.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = z.read(i)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f'Existing asset differs from archive: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as f:
                temporary = Path(f.name)
                f.write(content)
            os.replace(temporary, target)
    return destination / 'Wheel_Robot 0.8/urdf/Wheel_Robot.urdf'

def vector(element, key, default='0 0 0'):
    return np.fromstring(element.get(key, default) if element is not None else default, sep=' ')

def transform(origin):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler('xyz', vector(origin, 'rpy')).as_matrix()
    t[:3, 3] = vector(origin, 'xyz')
    return t

class Asset:
    def __init__(self, urdf):
        self.path = Path(urdf)
        self.root = ET.parse(urdf).getroot()
        self.links = {e.get('name'): e for e in self.root.findall('link')}
        self.joints = {e.get('name'): e for e in self.root.findall('joint')}
        self.parents = {j.find('child').get('link'): j for j in self.joints.values()}
        roots = set(self.links) - self.parents.keys()
        if len(roots) != 1 or len(self.parents) != len(self.joints):
            raise ValueError('Expected one rooted tree, unique child per joint')
        self.base = roots.pop()
        for link in self.links:
            self.chain(link)  # also rejects cycles/disconnected components

    def chain(self, link):
        chain, visited = [], set()
        while link != self.base:
            if link in visited or link not in self.parents:
                raise ValueError('Invalid kinematic tree')
            visited.add(link)
            j = self.parents[link]
            chain.append(j.get('name'))
            link = j.find('parent').get('link')
        return chain[::-1]

    def mesh_path(self, uri):
        # All supplied filenames are unique. Never silently fetch a missing resource.
        name = uri.rsplit('/', 1)[-1]
        matches = list((self.path.parent.parent / 'meshes').glob(name))
        if len(matches) != 1:
            raise ValueError(f'Missing or ambiguous resource: {uri}')
        return matches[0]

    def mesh(self, element):
        e = element.find('geometry/mesh')
        if e is None:
            raise ValueError('This asset adapter expects mesh geometry')
        mesh = trimesh.load_mesh(self.mesh_path(e.get('filename')))
        mesh.apply_scale(vector(e, 'scale', '1 1 1'))
        mesh.apply_transform(transform(element.find('origin')))
        if not np.isfinite(mesh.vertices).all():
            raise ValueError('Nonfinite mesh')
        return mesh

    def audit(self):
        out = dict(robot=self.root.get('name'), urdf_sha256=sha256(self.path), root=self.base,
                   joints=[], links=[], meshes=[], source_files=[])
        for name, j in self.joints.items():
            limit = j.find('limit')
            out['joints'].append(dict(name=name, type=j.get('type'),
                parent=j.find('parent').get('link'), child=j.find('child').get('link'),
                origin_xyz_m=vector(j.find('origin'), 'xyz').tolist(),
                origin_rpy_rad=vector(j.find('origin'), 'rpy').tolist(),
                axis_joint_frame=vector(j.find('axis'), 'xyz', '1 0 0').tolist(),
                limit=None if limit is None else {k: float(v) for k,v in limit.attrib.items()}))
        for name, l in self.links.items():
            i = l.find('inertial')
            entry = dict(name=name, chain=self.chain(name), visual_count=len(l.findall('visual')),
                         collision_count=len(l.findall('collision')), inertial=None)
            if i is not None:
                a = i.find('inertia').attrib
                mat = np.array([[float(a['ixx']),float(a['ixy']),float(a['ixz'])],
                                [float(a['ixy']),float(a['iyy']),float(a['iyz'])],
                                [float(a['ixz']),float(a['iyz']),float(a['izz'])]])
                eig = np.linalg.eigvalsh(mat)
                entry['inertial'] = dict(mass_kg=float(i.find('mass').get('value')),
                    origin_xyz_m=vector(i.find('origin'),'xyz').tolist(),
                    origin_rpy_rad=vector(i.find('origin'),'rpy').tolist(), tensor_kg_m2=mat.tolist(),
                    principal_kg_m2=eig.tolist(), positive_semidefinite=bool(eig[0]>=-1e-12),
                    triangle_inequality=bool(eig[2]<=eig[0]+eig[1]+1e-12))
            out['links'].append(entry)
            for kind in ['visual','collision']:
                for e in l.findall(kind):
                    me = e.find('geometry/mesh'); p = self.mesh_path(me.get('filename'))
                    m = self.mesh(e)
                    out['meshes'].append(dict(link=name, kind=kind, uri=me.get('filename'),
                        resolved_relative_path=str(p.relative_to(self.path.parent.parent)), sha256=sha256(p),
                        resolution_rule='unique supplied basename; original URI preserved',
                        scale=vector(me,'scale','1 1 1').tolist(), triangles=len(m.faces),
                        watertight=bool(m.is_watertight), bounds_link_frame_m=m.bounds.tolist()))
        for p in sorted(self.path.parent.parent.rglob('*')):
            if p.is_file():
                out['source_files'].append(dict(path=str(p.relative_to(self.path.parent.parent)),
                                                size=p.stat().st_size,sha256=sha256(p)))
        out['declared_mass_sum_kg'] = sum(e['inertial']['mass_kg'] for e in out['links'] if e['inertial'])
        return out

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
