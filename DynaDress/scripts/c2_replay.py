"""Calibrate validation-only noise, then causal nonoverlapping C2 replay.

No model fitting. Oracle truth is used only for noise calibration and scoring,
never passed to a belief update or candidate selector.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.inference.belief import JointBelief, hypotheses, SegmentLikelihood
from dynadress.inference.predictor import FrozenPredictor, CHECKPOINT_SHA256
from dynadress.inference.segments import summary, phase
from dynadress.human.geometry import axial_progress
QREF=np.array([0.,1.,0.,.35],dtype=np.float32)


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def get_history(d,t,h=8):
    return d['state'][t-h+1:t+1],d['action'][t-h:t]


def calibrate(predictor,dataset,index,output,steps=5):
    residual=[];ids=[];g=predictor.schema['groups']
    for row in index:
        if row['split']!='validation':continue
        d=dict(np.load(dataset/row['file']));ids.append(row['episode_id']);base=d['state'][:8,-12:].mean(0)
        for t in range(8,len(d['state'])-steps,steps):
            h,p=get_history(d,t)
            mu=float(d['condition'][8]);grid=np.array([[d['condition'][0],d['condition'][4],mu]])
            pred,_=predictor.observe(h,p,d['action'][t:t+steps],grid,QREF,base)
            y=summary(h[-1],d['state'][t+1:t+steps+1],g,base)
            residual.append(y-pred[0])
    residual=np.asarray(residual)
    # Median absolute error about zero retains model bias in the uncertainty.
    # Student-t4 median |t| is ~0.740697. Floors express numerical/model validity,
    # not a hypothetical physical sensor calibration.
    floor=np.r_[[.0005]*4,[.005]*4,.00025,[.2]*3,[.02]*3,[.2]*3,[.02]*3]
    scale=np.maximum(np.median(abs(residual),axis=0)/.740697,floor)
    cfg=dict(version='c2.student_t.v1',df=4.,scale=scale.tolist(),floors=floor.tolist(),
        source='old validation episodes only; oracle residuals, nonoverlapping segments', horizon_steps=steps,
        validation_episodes=ids,segments=len(residual),checkpoint_sha256=CHECKPOINT_SHA256,
        split_sha256=hashlib.sha256((dataset/'split_groups.json').read_bytes()).hexdigest(),
        channel_order='endpoint q4, dq4, axial displacement1, baseline-subtracted mean wrist wrench12',
        likelihood='multivariate Student-t df4 per q/dq/relative/load group; diagonal scales, shared group radial noise',
        residual_median=np.median(residual,axis=0).tolist())
    dump(output/'calibration.json',cfg);np.savez_compressed(output/'validation_residuals.npz',residual=residual)
    return cfg


def replay_episode(d,predictor,likelihood,grid):
    g=predictor.schema['groups'];base=d['state'][:8,-12:].mean(0)
    beliefs={name:JointBelief(grid) for name in ('human_only','full')}
    rows=[];errors={};geodiff=axial_progress(d['state'][:,:4],d['state'][:,slice(*g['cuff_center'])])-d['state'][:,g['progress'][0]]
    # Oracle is a separate scoring index; grid does not depend on episode truth.
    truth=np.array([d['condition'][0],d['condition'][4],d['condition'][8]])
    expanded=np.vstack([grid,[12.,1.25,.3],truth])
    for t in range(8,len(d['state'])-10,5):
        h,p=get_history(d,t);actions=d['action'][t:t+10]
        predictions=predictor.rollout(h,p,actions,expanded,QREF)
        pred=predictions[:len(grid)]
        ph=phase(h[-1],d['state'][t+1:t+6],g,QREF,base)
        y=summary(h[-1],d['state'][t+1:t+6],g,base)
        means=summary(h[-1],pred[:,:5],g,base)
        before={k:b.summary() for k,b in beliefs.items()}
        # Predictions are scored using the belief at t, before y(t+.25) arrives.
        for mode in ('middle','oracle','human_only','full'):
            yp=(predictions[-2] if mode=='middle' else predictions[-1] if mode=='oracle' else np.einsum('n,ntd->td',beliefs[mode].weights,pred))
            for n in (5,10):
                e=yp[n-1]-d['state'][t+n]
                vals=dict(q=e[:4],dq=e[4:8],progress=e[g['progress'][0]:g['progress'][1]],load=e[-12:])
                for name,value in vals.items():errors.setdefault(f'{mode}/{n*.05:.2f}/{name}',[]).extend(value.tolist())
        for mode,b in beliefs.items(): b.update(y,means,likelihood,ph,human_only=mode=='human_only')
        rows.append(dict(anchor_index=t,segment_start_s=float(d['time'][t]),observed_until_s=float(d['time'][t+5]),
                         phase=ph,observation=y.tolist(),before=before,after={k:b.summary() for k,b in beliefs.items()}))
    metrics={key:float(np.sqrt(np.mean(np.asarray(v)**2))) for key,v in errors.items()}
    return rows,dict(prediction_rmse=metrics,geometry_progress_consistency_rmse_m=float(np.sqrt(np.mean(geodiff**2))),
                     truth_for_scoring_only=truth.tolist(),final={k:b.summary() for k,b in beliefs.items()},segments=len(rows))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/m2/dataset')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/c2/replay');p.add_argument('--device',default='cpu')
    p.add_argument('--calibrate',action='store_true');p.add_argument('--fixed-mu',type=float)
    p.add_argument('--calibrate-only',action='store_true');p.add_argument('--calibration-steps',type=int,default=5)
    p.add_argument('--split',default='test');p.add_argument('--limit',type=int)
    a=p.parse_args();import torch;torch.set_num_threads(2)
    started=time.time();predictor=FrozenPredictor(ROOT/'artifacts/m21/model/best.pt',a.device)
    index=json.loads((a.dataset/'index.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
    calpath=ROOT/('artifacts/c2/calibration/calibration.json' if a.calibration_steps==5 else 'artifacts/c2/probe_calibration/calibration.json')
    cal=calibrate(predictor,a.dataset,index,calpath.parent,a.calibration_steps) if a.calibrate or not calpath.exists() else json.loads(calpath.read_text())
    if a.calibrate_only:return
    likelihood=SegmentLikelihood(cal['scale'],cal['df']);grid=hypotheses(a.fixed_mu)
    selected=[r for r in index if r['split']==a.split and (a.fixed_mu is None or abs(r['friction']-a.fixed_mu)<1e-6)]
    if a.limit:selected=selected[:a.limit]
    report=[]
    for r in selected:
        d=dict(np.load(a.dataset/r['file']));rows,metrics=replay_episode(d,predictor,likelihood,grid)
        dump(a.output/(r['episode_id']+'.json'),dict(hypotheses=grid.tolist(),updates=rows,metrics=metrics))
        report.append(dict(episode_id=r['episode_id'],**metrics));print('C2_REPLAY',r['episode_id'],metrics['segments'],flush=True)
    dump(a.output/'metrics.json',dict(episodes=report,hypotheses=grid.tolist(),checkpoint_sha256=CHECKPOINT_SHA256,
         calibration=str(calpath.relative_to(ROOT)),wall_seconds=time.time()-started,
         note='Old test set is regression. Predictions use belief before future observation. No template or ID input.'))
    print('C2_REPLAY_COMPLETE',len(report),flush=True)
if __name__=='__main__':main()
