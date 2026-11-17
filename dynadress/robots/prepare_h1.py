"""Audit supplied AR5 whole-body asset; author a derived fixed-base task URDF."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

EXPECTED = '5e40a5558cf3ed0dd59d1e9a08fa38944ddbb7c48e1612d0b7010a39cbeeb977'

def prepare(root):
    root = Path(root).resolve()
    archive = root / 'source.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Supplied archive hash differs from audited asset')
    import zipfile
    if not (root / 'original').exists():
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (root / 'original' / member.filename).resolve()
                if not target.is_relative_to(root / 'original'):
                    raise ValueError('Unsafe archive path')
            z.extractall(root / 'original')
    files=list((root/'original').rglob('*.urdf'))
    if len(files)!=1: raise ValueError('Expected one supplied URDF')
    source=files[0]; tree=ET.parse(source); robot=tree.getroot()
    out=root/'generated'; out.mkdir(exist_ok=True)
    report={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'archive_sha256':EXPECTED, 'identity':'user-supplied Wheel_Robot 0.8 / dual AR5; not serial-number certified',
            'mesh_resolution':[], 'joints':[], 'inertial_warnings':[],
            'task_edits':['non-arm joints fixed at URDF zero','simulation-only parallel jaw tools added; user authorized'],
            'source_gripper_present':False}
    for mesh in robot.findall('.//mesh'):
        raw=mesh.get('filename'); name=raw.rsplit('/',1)[-1]
        candidates=list((root/'original').rglob(name))
        if len(candidates)!=1: raise ValueError(f'Mesh unresolved/ambiguous: {raw}')
        mesh.set('filename',str(candidates[0]))
        report['mesh_resolution'].append({'original':raw,'resolved':str(candidates[0]),'sha256':hashlib.sha256(candidates[0].read_bytes()).hexdigest()})
    for link in robot.findall('link'):
        inert=link.find('inertial')
        if inert is None:
            report['inertial_warnings'].append({'link':link.get('name'),'reason':'missing inertial; fixed-link geometry retained'})
            continue
        mass=float(inert.find('mass').get('value')); a=inert.find('inertia'); get=lambda k:float(a.get(k))
        mat=np.array([[get('ixx'),get('ixy'),get('ixz')],[get('ixy'),get('iyy'),get('iyz')],[get('ixz'),get('iyz'),get('izz')]])
        eig=np.linalg.eigvalsh(mat)
        if mass<=0 or min(eig)<=0: report['inertial_warnings'].append({'link':link.get('name'),'mass':mass,'principal_inertia':eig.tolist()})
    for j in robot.findall('joint'):
        arm='ZY2_joint_' in j.get('name')
        report['joints'].append({'name':j.get('name'),'type_original':j.get('type'),'task_fixed':not arm,'limit':j.find('limit').attrib if j.find('limit') is not None else None})
        if not arm:
            j.set('type','fixed')
            for tag in ['limit','dynamics','mimic']:
                e=j.find(tag)
                if e is not None:j.remove(e)
    for side in ['L','R']:
        flange=f'AR5-5_08{side}-W4C1C9-ZY2_flan_link'
        palm=f'tool_{side}'
        for name,mass,size in [(palm,.2,[.025,.06,.04]),(palm+'_finger_a',.04,[.01,.008,.07]),(palm+'_finger_b',.04,[.01,.008,.07])]:
            link=ET.SubElement(robot,'link',name=name)
            inert=ET.SubElement(link,'inertial'); ET.SubElement(inert,'mass',value=str(mass)); ET.SubElement(inert,'origin',xyz='0 0 0')
            x,y,z=size; ET.SubElement(inert,'inertia',ixx=str(mass*(y*y+z*z)/12),iyy=str(mass*(x*x+z*z)/12),izz=str(mass*(x*x+y*y)/12),ixy='0',ixz='0',iyz='0')
            for kind in ['visual','collision']:
                shape=ET.SubElement(link,kind); geo=ET.SubElement(shape,'geometry'); ET.SubElement(geo,'box',size=' '.join(map(str,size)))
        fixed=ET.SubElement(robot,'joint',name=palm+'_mount',type='fixed'); ET.SubElement(fixed,'parent',link=flange); ET.SubElement(fixed,'child',link=palm); ET.SubElement(fixed,'origin',xyz='0 0 0.03')
        for k,sign in [('a',1),('b',-1)]:
            j=ET.SubElement(robot,'joint',name=palm+'_jaw_'+k,type='prismatic'); ET.SubElement(j,'parent',link=palm); ET.SubElement(j,'child',link=palm+'_finger_'+k); ET.SubElement(j,'origin',xyz=f'0 {sign*.012} .05'); ET.SubElement(j,'axis',xyz=f'0 {sign} 0'); ET.SubElement(j,'limit',lower='0',upper='.035',effort='15',velocity='.05')
    tree.write(out/'h1_task.urdf',encoding='utf-8',xml_declaration=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'urdf':str(out/'h1_task.urdf'),'source_sha256':report['source_sha256'],'source_gripper_present':False,'warnings':len(report['inertial_warnings'])}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');prepare(p.parse_args().root)
