"""Render C2 figures from saved metrics and measured sequences only."""
import argparse,gzip,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.inference.belief import JointBelief

def read_json(path):
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as f:return json.load(f)

COLORS={'passive':'#3465a4','fixed':'#c17d11','active':'#4e9a06'}


def save(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.pdf'));plt.close(fig)


def pair_curve(entries,out,pair,title,name):
    selected=[e for e in entries if e['pair_id']==pair]
    fig,axes=plt.subplots(3,2,figsize=(11,8),sharex=True)
    for e in selected:
        b=read_json(out/(e['episode_id']+'_beliefs.json'));up=b['updates'];truth=[e['K'],e['D'],e['mu']]
        grid=np.asarray(b['hypotheses']);times=[u['observed_until_s'] for u in up]
        probs=[];means=[]
        for u in up:
            s=u['after']['full'];w=np.asarray(s['joint']);c='low' if e['mu']==.1 else 'high'
            probs.append([w[np.isclose(grid[:,0],truth[0])].sum(),w[np.isclose(grid[:,1],truth[1])].sum(),s['effective_sliding_resistance'][c]])
            means.append(s['mean'])
        probs=np.asarray(probs);means=np.asarray(means);color=COLORS[e['mode']]
        for j in range(3):axes.flat[j].plot(times,probs[:,j],color=color,label=e['mode'])
        axes.flat[3].plot(times,means[:,0],color=color,label=e['mode'])
        data=dict(np.load(out/(e['episode_id']+'.npz')));s=data['state'];t=data['time']
        axes.flat[4].plot(t,s[:,72]*1000,color=color,label=e['mode'])
        force=np.linalg.norm(s[:,-12:].reshape(-1,2,6)[...,:3],axis=-1).sum(-1)
        axes.flat[5].plot(t,force,color=color,label=e['mode'])
        if e['trigger_s'] is not None:
            for ax in axes.flat:ax.axvspan(e['trigger_s'],e['trigger_s']+1.,color=color,alpha=.06)
    for ax,label in zip(axes.flat,['P(true K)','P(true D)','P(true resistance class)','Posterior mean K [Nm/rad]','Axial progress [mm]','Sum of wrist force norms [N]']):
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axes.flat[0].legend();axes.flat[3].axhline(selected[0]['K'],color='black',linestyle='--',alpha=.5)
    for ax in axes[-1]:ax.set_xlabel('Simulation time [s]')
    fig.suptitle(title+f" | {pair}, K={selected[0]['K']}, D={selected[0]['D']}, mu={selected[0]['mu']}",fontsize=11)
    save(fig,out,name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,default=ROOT/'artifacts/c2/evaluation');a=p.parse_args();out=a.evaluation
    r=json.loads((out/'metrics.json').read_text());entries=r['episodes']
    modes=('passive','fixed','active');fig,axs=plt.subplots(1,3,figsize=(11,3.5))
    for ax,key,label in zip(axs,('p_true_KD','p_true_resistance_class','K_abs_error'),('P(true K,D) after opportunity','P(true resistance class)','K absolute error [Nm/rad]')):
        values=[r['aggregate'][m][key] for m in modes];ax.bar(modes,values,color=[COLORS[m] for m in modes]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    save(fig,out,'paired_summary')
    # Show a genuinely correct active K/D result if one exists, without requiring
    # it to beat passive. The selection rule is saved, avoiding cherry-picked claims.
    act=[e for e in entries if e['mode']=='active'];correct=[e for e in act if e['final']['K_map_correct'] and e['final']['D_map_correct']]
    pool=correct or act
    def benefit(e):
        passive=next(x for x in entries if x['pair_id']==e['pair_id'] and x['mode']=='passive')
        return e['after']['p_true_KD']-passive['after']['p_true_KD']
    best=max(pool,key=benefit)
    confusing=min(act,key=lambda e:e['after']['p_true_resistance_class'])
    pair_curve(entries,out,best['pair_id'],('Correct final K/D case' if correct else 'Best available K/D case')+'; shaded probing window','identification_pair')
    pair_curve(entries,out,confusing['pair_id'],'Most confused active resistance-class case','ambiguous_pair')
    twins=[e for e in act if e['K']==confusing['K'] and e['D']==confusing['D']]
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True)
    for e in twins:
        b=read_json(out/(e['episode_id']+'_beliefs.json'));up=b['updates']
        t=[u['observed_until_s'] for u in up];label=f"true mu={e['mu']}"
        color='#3465a4' if e['mu']==.1 else '#cc0000'
        axes[0,0].plot(t,[u['after']['full']['effective_sliding_resistance']['high'] for u in up],color=color,label=label)
        axes[0,1].plot(t,[u['after']['full']['mean'][0] for u in up],color=color,label=label)
        d=dict(np.load(out/(e['episode_id']+'.npz')))
        axes[1,0].plot(d['time'],d['state'][:,72]*1000,color=color,label=label)
        force=np.linalg.norm(d['state'][:,-12:].reshape(-1,2,6)[...,:3],axis=-1).sum(-1)
        axes[1,1].plot(d['time'],force,color=color,label=label)
        for ax in axes.flat:ax.axvspan(e['trigger_s'],e['trigger_s']+1.,color=color,alpha=.06)
    for ax,label in zip(axes.flat,['P(high effective sliding resistance)','Posterior mean K [Nm/rad]','Axial progress [mm]','Sum wrist force norms [N]']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0,0].legend();axes[1,0].set_xlabel('Simulation time [s]');axes[1,1].set_xlabel('Simulation time [s]')
    fig.suptitle(f"Ambiguous friction conditions: active, K={confusing['K']}, D={confusing['D']}")
    save(fig,out,'ambiguous_conditions')
    (out/'figure_selection.json').write_text(json.dumps(dict(identification_pair=best['pair_id'],ambiguous_pair=confusing['pair_id'],correct_final_KD_case_available=bool(correct),
        rule='among correct final active K/D MAP cases, max matched-time p(true KD) gain over passive; ambiguity=min active p(true class)'),indent=2))
    # Actual human excitation must be checked independently of belief entropy.
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for col,pair in enumerate((confusing['pair_id'],best['pair_id'])):
        e=next(e for e in act if e['pair_id']==pair)
        d=dict(np.load(out/(e['episode_id']+'.npz')))
        b=read_json(out/(e['episode_id']+'_beliefs.json'))
        axes[0,col].plot(d['time'],np.linalg.norm(d['state'][:,:4]-[0,1,0,.35],axis=-1)*1000)
        axes[0,col].axhline(2,color='gray',ls='--',label='2 mrad deviation')
        axes[0,col].set_title(f"Active {pair}: true K={e['K']}, D={e['D']}, mu={e['mu']}")
        axes[1,col].plot([u['observed_until_s'] for u in b['updates']],[u['after']['full']['mean'][0] for u in b['updates']],color=COLORS['active'])
        axes[1,col].axhline(e['K'],color='black',ls='--',label='True K')
        for ax in axes[:,col]:
            ax.axvspan(e['trigger_s'],e['trigger_s']+1.,color=COLORS['active'],alpha=.15)
            ax.grid(alpha=.2)
        axes[1,col].set_xlabel('Simulation time [s]')
    axes[0,0].set_ylabel('Actual human |q - q_ref| [mrad]')
    axes[1,0].set_ylabel('Posterior mean K [Nm/rad]')
    axes[0,0].legend();axes[1,0].legend()
    fig.suptitle('Proxy-triggered probe precedes measurable human motion')
    save(fig,out,'probe_observability')
    diag=json.loads((out/'mu_diagnosis.json').read_text())['pairs']
    fig,axs=plt.subplots(1,2,figsize=(10,3.7))
    for ax,stage in zip(axs,('probe','late_sliding')):
        rows=[x for x in diag if x['stage']==stage]
        x=[v['load']['observed_scale_rms'] for v in rows];y=[v['load']['model_intervention_scale_rms'] for v in rows]
        for mode in modes:
            ix=[i for i,v in enumerate(rows) if v['mode']==mode]
            ax.scatter(np.asarray(x)[ix],np.asarray(y)[ix],c=COLORS[mode],label=mode)
        ax.axhline(1,color='gray',ls='--');ax.axvline(1,color='gray',ls='--');ax.set_title(stage)
        ax.set_xlabel('Observed paired load difference / validation scale');ax.set_ylabel('Same-history model mu sensitivity / scale');ax.grid(alpha=.2)
    axs[0].legend();save(fig,out,'friction_response_diagnosis')
    # Frozen old split: episode macro averages, no future-assimilated scoring.
    old=json.loads((ROOT/'artifacts/c2/replay/metrics.json').read_text())['episodes']
    fig,axs=plt.subplots(1,2,figsize=(9,3.7))
    for ax,horizon in zip(axs,('0.25','0.50')):
        names=('middle','human_only','full','oracle')
        vals=[np.mean([e['prediction_rmse'][f'{m}/{horizon}/q'] for e in old])*1000 for m in names]
        ax.bar(names,vals,color=['gray','#729fcf','#4e9a06','#ad7fa8']);ax.set_title(horizon+' s');ax.set_ylabel('Human angle RMSE [mrad]');ax.grid(axis='y',alpha=.2)
    save(fig,out,'replay_prediction')
    print('C2_FIGURES_COMPLETE')
if __name__=='__main__':main()
