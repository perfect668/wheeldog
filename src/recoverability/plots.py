"""Regenerate every figure from saved CSV/NPZ; no optimization during plotting."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

COLORS=['#167d9a','#c78627','#8364a5','#539453','#bb4d56']


def load_rows(directory):
    with (Path(directory)/'metrics.csv').open() as f:
        return list(csv.DictReader(f))


def boundary_rows(rows):
    output=[]
    groups=sorted(set((r['method'],r['actuator'],float(r['horizon_s'])) for r in rows if r['family']=='sweep'))
    for method,name,horizon in groups:
        group=[r for r in rows if r['family']=='sweep' and r['method']==method and r['actuator']==name
               and float(r['horizon_s'])==horizon and float(r['delta_v_m_s']) >= 0]
        group=sorted(group,key=lambda r:float(r['delta_v_m_s']))
        success='RECOVERABLE' if method=='oracle' else 'RECOVERED'
        good=[float(r['delta_v_m_s']) for r in group if r['status']==success]
        low=max(good) if good else float('nan')
        following=[r for r in group if float(r['delta_v_m_s'])>low]
        high=float(following[0]['delta_v_m_s']) if following else float('nan')
        # A policy need not have a connected success set; flag holes rather than hide them.
        holes=sum(r['status']!=success for r in group if float(r['delta_v_m_s'])<low)
        errors=sum('ERROR' in r['status'] for r in group)
        output.append({'method':method,'actuator':name,'horizon_s':horizon,
                       'largest_recovered_grid_m_s':low,'next_grid_m_s':high,
                       'next_grid_status':following[0]['status'] if following else 'grid_end',
                       'holes_below_largest':holes,'errors_positive_half':errors})
    return output


def plot_all(directory):
    directory=Path(directory)
    rows=load_rows(directory)
    figures=directory/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':160,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),bbox_inches='tight')
        fig.savefig(figures/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    boundaries=boundary_rows(rows)
    if boundaries:
        with (directory/'boundaries.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(boundaries[0])); w.writeheader(); w.writerows(boundaries)
        oracle=[r for r in rows if r['family']=='sweep' and r['method']=='oracle']
        names=list(dict.fromkeys(r['actuator'] for r in oracle))
        hs=sorted(set(float(r['horizon_s']) for r in oracle))
        vs=sorted(set(float(r['delta_v_m_s']) for r in oracle))
        fig,axes=plt.subplots(1,len(names),figsize=(3.2*len(names),5),sharey=True,squeeze=False)
        cmap=ListedColormap(['#dce2e8','#167d9a','#e6a33c'])
        for ax,name in zip(axes[0],names):
            grid=np.full((len(vs),len(hs)),2.0)
            for r in oracle:
                if r['actuator']==name:
                    grid[vs.index(float(r['delta_v_m_s'])),hs.index(float(r['horizon_s']))]={'INFEASIBLE':0,'RECOVERABLE':1,'SOLVER_ERROR':2}[r['status']]
            ax.imshow(grid,origin='lower',aspect='auto',cmap=cmap,norm=BoundaryNorm([-.5,.5,1.5,2.5],3),
                      extent=[-.5,len(hs)-.5,vs[0]-(vs[1]-vs[0])/2,vs[-1]+(vs[1]-vs[0])/2])
            ax.set_xticks(range(len(hs)),[f'{h:g}' for h in hs]);ax.set_xlabel('Horizon (s)')
            ax.set_title(f'{name}\nerrors: {int(np.sum(grid==2))}');ax.axhline(0,color='white',lw=.6)
        axes[0,0].set_ylabel('Velocity impulse Δv (m/s)')
        caps=sorted(set(json.loads(r['actuator_json'])['force_max_N'] for r in oracle))
        fig.suptitle(f'Toy simulation · offline known-disturbance oracle · force bounds {caps} N',y=1.02)
        fig.legend(handles=[Patch(color=cmap(i),label=l) for i,l in enumerate(['Infeasible','Recoverable','Solver / validation error'])],
                   loc='lower center',ncol=3,bbox_to_anchor=(.5,-.07))
        save(fig,'recoverability_heatmaps')
        fig,ax=plt.subplots(figsize=(8,5))
        for color,name in zip(COLORS,names):
            b=sorted([r for r in boundaries if r['method']=='oracle' and r['actuator']==name],key=lambda r:r['horizon_s'])
            x=[r['horizon_s'] for r in b]; y=[r['largest_recovered_grid_m_s'] for r in b]
            ax.plot(x,y,'o-',color=color,label=name)
            ax.fill_between(x,y,[r['next_grid_m_s'] for r in b],color=color,alpha=.12)
        ax.set(xlabel='Horizon (s)',ylabel='Positive Δv boundary (m/s)',title='Toy simulation · oracle grid boundary\nshading: boundary bracket on a 0.04 m/s grid (not a confidence interval)')
        ax.legend(ncol=2);ax.grid(alpha=.2);save(fig,'recovery_boundaries')
        # Causal success patterns shown point by point, so no policy monotonicity is assumed.
        causal=[r for r in rows if r['family']=='sweep' and r['method']=='causal_pd']
        if causal:
            fig,ax=plt.subplots(figsize=(9,4.5))
            for i,name in enumerate(names):
                group=sorted([r for r in causal if r['actuator']==name],key=lambda r:float(r['delta_v_m_s']))
                for r in group:
                    ax.scatter(float(r['delta_v_m_s']),i,color=COLORS[i],marker='o' if r['status']=='RECOVERED' else 'x',s=28)
            ax.set_yticks(range(len(names)),names)
            ax.set(xlabel='Velocity impulse Δv (m/s)',title='Toy simulation · causal fixed PD at H=2 s\no = recovered; × = this policy did not recover')
            ax.grid(alpha=.2);save(fig,'causal_feedback')
        fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        statuses=['RECOVERABLE','INFEASIBLE','SOLVER_ERROR']
        bottom=np.zeros(len(names))
        for status,color in zip(statuses,['#167d9a','#dce2e8','#e6a33c']):
            vals=np.array([sum(r['status']==status for r in oracle if r['actuator']==name) /
                           sum(r['actuator']==name for r in oracle) for name in names])
            axes[0].bar(names,vals,bottom=bottom,label=status,color=color);bottom+=vals
        axes[0].set(ylabel='Fraction of actual solves',ylim=(0,1));axes[0].legend(fontsize=8)
        axes[0].tick_params(axis='x',rotation=25)
        for name,color in zip(names,COLORS):
            vals=np.array([float(r['elapsed_s'])*1000 for r in oracle if r['actuator']==name])
            axes[1].hist(vals,bins=30,histtype='step',label=name,color=color)
        axes[1].set(xlabel='End-to-end oracle time (ms)',ylabel='Cases');axes[1].legend(fontsize=8)
        fig.suptitle('Toy simulation · oracle status and CPU timing');save(fig,'solver_diagnostics')
    # Shared fixed samples: include recovered and impossible disturbances, never invent a failed oracle path.
    data=np.load(directory/'trajectories.npz',allow_pickle=False)
    examples=[]
    for family,value in [('sweep',.20),('sweep',.36),('sweep',.64),('future_impulse',None),('force_pulse',None)]:
        selection=[r for r in rows if r['family']==family and float(r['horizon_s'])==2.0
                   and r['actuator'] in ('A_fast','B_slow')
                   and (value is None or abs(float(r['delta_v_m_s'])-value)<1e-9)]
        if selection:
            examples.append((family,value,selection))
    for family,value,selection in examples:
        fig,axes=plt.subplots(4,1,figsize=(9,9),sharex=True)
        for r in selection:
            key=r['case_id']; state=data[key+'_state'];force=data[key+'_force'];u=data[key+'_command']
            label=f"{r['actuator']} / {r['method']} / {r['status']}"
            if not len(state):
                # Report an infeasible solve explicitly; it has no physical trajectory to plot.
                axes[0].plot([],[],color='none',label=label+' (no trajectory)')
                continue
            t=np.arange(len(force))*float(r['dt_s'])
            color=COLORS[0] if r['actuator']=='A_fast' else COLORS[4]
            style='-' if r['method']=='oracle' else '--'
            axes[0].plot(t,state[:,0],style,color=color,label=label)
            axes[1].plot(t,state[:,1],style,color=color)
            axes[2].plot(t,force,style,color=color)
            axes[3].step(t[:-1],u,where='post',linestyle=style,color=color)
        model=json.loads(selection[0]['model_json'])
        cap=max(json.loads(r['actuator_json'])['force_max_N'] for r in selection)
        for ax,bound in zip(axes,[model['x_safe_m'],model['v_abs_max_m_s'],cap,cap]):
            for s in (-1,1):ax.axhline(s*bound,color='gray',lw=.7)
            ax.grid(alpha=.15)
        for ax,bound in zip(axes[:2],[model['x_terminal_m'],model['v_terminal_m_s']]):
            ax.plot([2],[bound],marker='|',color='black');ax.plot([2],[-bound],marker='|',color='black')
        for ax,label in zip(axes,['x (m)','v (m/s)','Actual force f (N)','Issued command u (N)']):ax.set_ylabel(label)
        axes[0].legend(fontsize=7,loc='best');axes[-1].set_xlabel('Time (s)')
        fig.suptitle(f'Toy simulation · {family}'+('' if value is None else f' · Δv={value:.2f} m/s'))
        fig.text(.5,.01,'Curves after path-limit violation are mathematical extrapolation outside the evaluation domain.',ha='center',fontsize=8)
        save(fig,'trajectory_'+family+('' if value is None else f'_{value:.2f}'))
    data.close()
