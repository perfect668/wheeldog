"""Generate figures only from saved M1 records and checksum-verified assets."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .assets import Asset,unpack
from .model import Robot,TIPS
from .collision import CollisionChecker


def display_bounds(records,robot,checker):
    points=[]
    for r in records:
        frames=robot.fk(np.array(r['q_rad']),r['base_xy_yaw'])
        for n,h in checker.hulls.items():
            t=frames[n];points.append(h.vertices@t[:3,:3].T+t[:3,3])
    all_points=np.concatenate(points)
    return np.array([all_points.min(axis=0)-.08,all_points.max(axis=0)+.08])


def render_robot(ax,robot,checker,q,base,targets=None,title='',bounds=None):
    frames=robot.fk(np.array(q),base)
    for n,h in checker.hulls.items():
        t=frames[n]; vertices=h.vertices@t[:3,:3].T+t[:3,3]
        color='#3c8dad' if '08L' in n else '#df934a' if '08R' in n else '#a8b3bd'
        collection=Poly3DCollection(vertices[h.faces],facecolor=color,edgecolor='none',alpha=.5)
        ax.add_collection3d(collection)
    for n in TIPS:
        t=frames[n]
        for i,color in enumerate(['r','g','b']):
            end=t[:3,3]+t[:3,i]*.08
            ax.plot(*np.stack([t[:3,3],end]).T,color=color,lw=1.2)
    if targets is not None:
        for target in targets: ax.scatter(*np.array(target)[:3,3],color='black',marker='x',s=25)
    if bounds is None: bounds=np.array([[-.65,-.95,-.35],[1.,.95,1.9]])
    ax.set(xlim=bounds[:,0],ylim=bounds[:,1],zlim=bounds[:,2],title=title,
           xlabel='world x [m]',ylabel='world y [m]',zlabel='world z [m]')
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]: axis.label.set_fontsize(8);axis.labelpad=1
    ax.set_box_aspect(bounds[1]-bounds[0]);ax.view_init(elev=19,azim=-49);ax.tick_params(labelsize=6)


def generate(output):
    out=Path(output);figs=out/'figures';figs.mkdir(exist_ok=True)
    read=lambda n:json.loads((out/n).read_text())
    import yaml
    config=yaml.safe_load((out/'config.yaml').read_text())
    asset=Asset(unpack(config['archive'],config['asset_directory']));robot=Robot(asset);checker=CollisionChecker(asset)
    candidates=read('candidates.json'); targets=read('targets.json')['world_targets']
    accepted=[r for r in candidates if r['status']=='ACCEPTED_CONDITIONAL']
    selected=(accepted[:1]+accepted[len(accepted)//2:len(accepted)//2+1]+accepted[-1:]) if accepted else []
    bounds=display_bounds(selected,robot,checker) if selected else None
    fig=plt.figure(figsize=(14,5.5),layout='constrained')
    for i,r in enumerate(selected):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        b=r['base_xy_yaw']
        render_robot(ax,robot,checker,r['q_rad'],b,targets,
                     f"{r['id']}; base=({b[0]:.3f}, {b[1]:.3f}) m, yaw={b[2]:.3f} rad",bounds=bounds)
    fig.suptitle('URDF kinematic configurations: identical world flange poses\nHull display only; collision tests use supplied mesh surfaces. No floor/person model.',fontsize=11)
    fig.savefig(figs/'redundant_configurations.png',dpi=180,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    random=[r for r in read('random_validation.json') if r['status']=='ACCEPTED_CONDITIONAL']
    bounds=display_bounds(random[:10],robot,checker) if random else None
    fig=plt.figure(figsize=(16,10),layout='constrained')
    for i,r in enumerate(random[:10]):
        ax=fig.add_subplot(2,5,i+1,projection='3d')
        render_robot(ax,robot,checker,r['q_rad'],r['base_xy_yaw'],title=r['id'],bounds=bounds)
    fig.suptitle('10 random configurations passing URDF limits + supplied mesh surface checks\nConditional geometry only; omitted floor/environment means these are not hardware operating poses.',fontsize=12)
    fig.savefig(figs/'random_validation.png',dpi=160,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    d=read('differential_checks.json');steps=sorted(set(r['epsilon_rad'] for r in d),reverse=True)
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for key,label in [('jacobian_linear_max_abs','linear J [m/rad]'),('jacobian_angular_max_abs','angular J [rad/rad]')]:
        axes[0].loglog(steps,[max(r[key] for r in d if r['epsilon_rad']==s) for s in steps],'o-',label=label)
    axes[0].set(xlabel='central difference step [rad]',ylabel='max absolute error',title='Both arms; all 24 joint columns');axes[0].legend(fontsize=8)
    for status,color in [('ACCEPTED_CONDITIONAL','#438a65'),('REJECTED','#c56952'),('NUMERICAL_ERROR','#8e71a9')]:
        axes[1].bar(status.replace('_','\n'),sum(r['status']==status for r in candidates),color=color)
    axes[1].set(ylabel='candidate count',title='All IK attempts');axes[1].tick_params(labelsize=7)
    if accepted:
        b=np.array([r['base_xy_yaw'] for r in accepted]);s=axes[2].scatter(b[:,0],b[:,1],c=b[:,2],cmap='coolwarm',s=35)
        fig.colorbar(s,ax=axes[2],label='base yaw [rad]')
    axes[2].set(xlabel='base x [m]',ylabel='base y [m]',title='Accepted static base placements')
    fig.suptitle('M1 static numerical experiment; supplied URDF, not physical H1 validation')
    fig.savefig(figs/'kinematics_diagnostics.png',dpi=180,bbox_inches='tight',pad_inches=.2);plt.close(fig)
    print(figs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');generate(p.parse_args().output)
