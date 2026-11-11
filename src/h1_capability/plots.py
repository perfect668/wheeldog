"""Rebuild M1-C figures from saved numerical records; no synthetic results."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import DIRECTIONS

LABELS=['+X front','−X back','+Y left','−Y right','+Z up','−Z down']
COLORS=['#3478a3','#d48b36','#459479','#b85e78','#7764aa','#8e794d']


def generate(output):
    out=Path(output);figs=out/'figures';figs.mkdir(exist_ok=True)
    read=lambda p:json.loads((out/p).read_text())
    records=[r for r in read('force_and_instantaneous.json') if r['status']=='ACCEPTED']
    reach=read('reachability.json');pairs=read('selected_pairs.json');summary=read('summary.json')
    lookup={(r['candidate'],r['horizon_s'],r['mobile'],r['direction']):r for r in reach}
    save=lambda fig,name:fig.savefig(figs/name,dpi=170,bbox_inches='tight',pad_inches=.2)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13,9),layout='constrained')
    for ax,kind,title in zip(axes,['unloaded','residual'],['Unloaded net ray force','Increment after 30 N upward per hand']):
        data=np.array([[r['force']['nominal'][kind][d]['capacity_n'] for d in DIRECTIONS] for r in records])
        im=ax.imshow(data,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='net force [N]')
        ax.set(xticks=range(6),xticklabels=LABELS,yticks=range(len(records)),yticklabels=[r['id'] for r in records],title=title)
        ax.tick_params(axis='y',labelsize=6);ax.tick_params(axis='x',rotation=30)
    fig.suptitle('Same world handles, nominal arm torque budgets; equal force increments\nArm-budget relaxation: shared torso loads recorded without unknown limits; no gravity or stability')
    save(fig,'force_capabilities.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for k,(tag,label) in enumerate([('urdf','URDF budget'),('nominal_half','0.5 × nominal'),('nominal','Nominal')]):
        for ax,kind in zip(axes,['unloaded','residual']):
            values=np.array([[r['force'][tag][kind][d]['capacity_n'] for d in DIRECTIONS] for r in records])
            ax.errorbar(np.arange(6)+(k-1)*.13,np.median(values,axis=0),
                yerr=np.array([np.median(values,axis=0)-values.min(axis=0),values.max(axis=0)-np.median(values,axis=0)]),
                fmt='o',capsize=3,label=label)
            ax.set(xticks=range(6),xticklabels=LABELS,ylabel='net ray force [N]',title=kind)
            ax.tick_params(axis='x',rotation=30)
    axes[0].legend(fontsize=8);fig.suptitle('Torque-budget sensitivity across all 42 configurations; median and full range')
    save(fig,'torque_sensitivity.png');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,d,label in zip(axes.flat,DIRECTIONS,LABELS):
        for mode,color,offset in [(False,'#3478a3',-.11),(True,'#d48b36',.11)]:
            data=[[100*r['achieved_displacement_m'] for r in reach if r['direction']==d and r['mobile']==mode and r['horizon_s']==h and r['achieved_displacement_m'] is not None] for h in [.2,.3,.5]]
            bp=ax.boxplot(data,positions=np.arange(3)+offset,widths=.19,patch_artist=True,manage_ticks=False,
                boxprops={'facecolor':color,'alpha':.55},medianprops={'color':'black'})
            ax.plot([],[],color=color,label='Mobile ∪ fixed' if mode else 'Fixed base')
        ax.set(xticks=range(3),xticklabels=['0.2','0.3','0.5'],xlabel='horizon [s]',ylabel='achieved displacement [cm]',title=label)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('42 configurations, synchronized fixed-orientation handles, rest to rest\nVerified local path lower bounds; shared acceleration/delay assumptions, sampled geometry')
    save(fig,'finite_time_population.png');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,d,label in zip(axes.flat,DIRECTIONS,LABELS):
        x=[r['force']['nominal']['residual'][d]['capacity_n'] for r in records]
        for mobile,marker,color in [(False,'o','#3478a3'),(True,'^','#d48b36')]:
            y=[100*lookup[(r['id'],.3,mobile,d)]['achieved_displacement_m'] for r in records]
            rho=next(r['spearman_rho'] for r in summary['correlation'] if r['mobile']==mobile and r['direction']==d)
            ax.scatter(x,y,s=18,marker=marker,color=color,label=f'{"Mobile" if mobile else "Fixed"}: ρ={rho:.2f}')
        ax.set(xlabel='current residual ray force [N]',ylabel='0.3 s path bound [cm]',title=label);ax.legend(fontsize=7)
    fig.suptitle('Current arm force budget versus future handle motion; descriptive 42-posture sample\nCorrelation is task- and planner-dependent; no population significance or hardware claim')
    save(fig,'force_vs_motion.png');plt.close(fig)

    fig,axes=plt.subplots(len(pairs),3,figsize=(15,3.7*len(pairs)),squeeze=False,layout='constrained')
    for i,p in enumerate(pairs):
        for candidate,color in [(p['a'],'#3478a3'),(p['b'],'#d48b36')]:
            r=next(r for r in records if r['id']==candidate)
            for kind,style in [('unloaded','--'),('residual','-')]:
                axes[i,0].plot(range(6),[r['force']['nominal'][kind][d]['capacity_n'] for d in DIRECTIONS],style+'o',ms=3,color=color,label=f'{candidate} {kind}')
            for j,mobile in enumerate([False,True]):
                axes[i,j+1].bar(np.arange(6)+(-.17 if candidate==p['a'] else .17),
                    [100*lookup[(candidate,.3,mobile,d)]['achieved_displacement_m'] for d in DIRECTIONS],
                    width=.32,color=color,label=candidate)
        for ax in axes[i]:ax.set(xticks=range(6),xticklabels=LABELS);ax.tick_params(axis='x',rotation=30);ax.legend(fontsize=6)
        axes[i,0].set(ylabel='net force [N]',title=f'{p["a"]} / {p["b"]}; max force gap {100*p["max_relative_gap"]:.2f}%')
        axes[i,1].set(ylabel='path bound [cm]',title='0.3 s, fixed base')
        axes[i,2].set(ylabel='path bound [cm]',title='0.3 s, mobile ∪ fixed')
    fig.suptitle('Top three disjoint descriptive pairs by mean absolute gap over all six fixed-base rays\nOutcome-selected examples, identical per-condition parameters; not independent validation')
    save(fig,'matched_pairs.png');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for mobile,ax in zip([False,True],axes):
        for h,color in zip([.2,.3,.5],COLORS):
            values=[100*x['gap_m'] for p in read('pair_comparisons.json') for x in p['comparisons'] if x['horizon_s']==h and x['mobile']==mobile and x['gap_m'] is not None]
            x=np.sort(values);ax.plot(x,np.arange(1,len(x)+1)/len(x),color=color,label=f'H={h}s')
        ax.axvline(1,color='gray',ls=':',label='1 cm absolute criterion')
        ax.set(xlabel='absolute pair gap [cm]',ylabel='empirical fraction',title='Mobile ∪ fixed' if mobile else 'Fixed base');ax.legend(fontsize=8)
    fig.suptitle('All 185 force-matched pairs × six directions; overlapping, dependent pairs\nAbsolute gap of achieved lower bounds; ratio criterion assessed separately')
    save(fig,'all_pair_gaps.png');plt.close(fig)

    if (out/'sensitivity.json').exists():
        sensitivity=read('sensitivity.json');ids=sorted({x for p in pairs for x in [p['a'],p['b']]})
        variants=['main']+sorted(set(r['experiment'] for r in sensitivity))
        fig,axes=plt.subplots(2,len(ids),figsize=(3*len(ids),9),squeeze=False,layout='constrained')
        vmax=max(100*r['achieved_displacement_m'] for r in sensitivity if r['achieved_displacement_m'] is not None)
        for col,candidate in enumerate(ids):
            for row,mobile in enumerate([False,True]):
                data=np.full((len(variants),6),np.nan)
                for i,variant in enumerate(variants):
                    for j,d in enumerate(DIRECTIONS):
                        r=lookup[(candidate,.3,mobile,d)] if variant=='main' else next((r for r in sensitivity if r['candidate']==candidate and r['experiment']==variant and r['mobile']==mobile and r['direction']==d),None)
                        if r is not None and r['achieved_displacement_m'] is not None:data[i,j]=100*r['achieved_displacement_m']
                ax=axes[row,col];im=ax.imshow(data,aspect='auto',vmin=0,vmax=vmax,cmap='viridis')
                ax.set(xticks=range(6),xticklabels=LABELS,yticks=range(len(variants)),
                    yticklabels=variants if col==0 else [],title=f'{candidate}, {"mobile" if mobile else "fixed"}')
                ax.tick_params(labelsize=7,axis='x',rotation=90)
        fig.colorbar(im,ax=list(axes.flat),label='0.3 s path bound [cm]',shrink=.7)
        fig.suptitle('Sensitivity: every selected-pair member, all six rays; blank = not run or initial rejection\nNo pair-specific actuator parameters; inverse-policy and dt variants expose planner sensitivity')
        save(fig,'motion_sensitivity.png');plt.close(fig)
    print('figures:',figs)


def render_pairs(output):
    import yaml
    from h1_kinematics.plots import render_robot,display_bounds
    from .experiment import setup
    out=Path(output);config=yaml.safe_load((out/'config.yaml').read_text());_,model,checker=setup(config)
    pairs=json.loads((out/'selected_pairs.json').read_text());records=json.loads((out/'force_and_instantaneous.json').read_text())
    chosen=[next(r for r in records if r['id']==x) for p in pairs for x in [p['a'],p['b']]]
    bounds=display_bounds(chosen,model.robot,checker)
    fig=plt.figure(figsize=(10,5*len(pairs)),layout='constrained')
    for i,r in enumerate(chosen):
        ax=fig.add_subplot(len(pairs),2,i+1,projection='3d');b=r['base_xy_yaw']
        render_robot(ax,model.robot,checker,r['q_rad'],b,r['handle_world_poses'],
            title=f'{r["id"]}; base ({b[0]:.3f}, {b[1]:.3f}) m, yaw {b[2]:.3f} rad',bounds=bounds)
    fig.suptitle('Selected URDF postures at the same world handles\nHull illustration, original mesh checks; frozen torso during arm/base motion, no floor/person model')
    fig.savefig(out/'figures'/'pair_configurations.png',dpi=150,bbox_inches='tight',pad_inches=.2);plt.close(fig)


def trajectory_example(output):
    """Plot the saved motions of pair 1, largest gap ray, with actual FK."""
    import yaml
    from scipy.interpolate import CubicHermiteSpline
    from .experiment import setup
    out=Path(output);config=yaml.safe_load((out/'config.yaml').read_text());_,model,_=setup(config)
    pair=json.loads((out/'selected_pairs.json').read_text())[0]
    comparison=max((x for x in pair['comparisons'] if x['horizon_s']==.3 and not x['mobile']),key=lambda x:x['gap_m'])
    direction=comparison['direction'];d=DIRECTIONS[direction]
    reach=json.loads((out/'reachability.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
    for col,candidate in enumerate([pair['a'],pair['b']]):
        row=next(r for r in reach if r['candidate']==candidate and r['horizon_s']==.3 and not r['mobile'] and r['direction']==direction)
        with np.load(row['path_file'],allow_pickle=False) as data:
            spline=CubicHermiteSpline(data['times_s'],data['coordinates'],data['slopes'])
            q0=data['q0_rad'];base=data['base0_xy_yaw'];initial=data['initial_handle_poses']
        times=np.linspace(0,.3,301);positions=[]
        for t in times:
            q=q0.copy();q[model.active]=spline(t)
            positions.append([100*np.dot(p[:3,3]-p0[:3,3],d) for p,p0 in zip(model.poses(q,base),initial)])
        positions=np.array(positions)
        for side,label,style in [(0,'Left handle','-'),(1,'Right handle','--')]:
            axes[0,col].plot(times,positions[:,side],style,label=label)
        axes[0,col].set(title=f'{candidate}: {100*row["achieved_displacement_m"]:.2f} cm',ylabel='handle translation [cm]');axes[0,col].legend()
        velocity=abs(spline(times,1))/model.velocity
        acceleration=abs(spline(times,2))/config['profiles']['nominal']['joint_acceleration_rad_s2']
        for ax,values,label in [(axes[1,col],velocity,'|joint velocity| / limit'),(axes[2,col],acceleration,'|joint acceleration| / assumed limit')]:
            ax.plot(times,values,alpha=.55,lw=.7);ax.axhline(1,color='black',ls=':');ax.axvline(.04,color='gray',ls=':')
            ax.set(ylabel=label,xlabel='time [s]',ylim=(-.03,1.08))
    fig.suptitle(f'Saved fixed-base paths, pair 1; {direction} is its largest-gap ray (descriptive selection)\nBoth handles translate together; 14 arm joints, same limits and 40 ms delay')
    fig.savefig(out/'figures'/'trajectory_example.png',dpi=170,bbox_inches='tight',pad_inches=.2);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('--postures',action='store_true')
    args=parser.parse_args();generate(args.output)
    if args.postures:
        render_pairs(args.output)
        trajectory_example(args.output)
