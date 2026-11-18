"""Paired observed friction response vs frozen-model property sensitivity.

One seed, one trajectory per mode/condition cannot estimate simulator variability.
Standardized separation is a diagnostic against old validation model residuals,
not a statistical proof of physical identifiability.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.inference.predictor import FrozenPredictor
from dynadress.inference.segments import summary
from scripts.c2_replay import QREF,get_history,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,default=ROOT/'artifacts/c2/evaluation');a=p.parse_args()
    import torch;torch.set_num_threads(2)
    report=json.loads((a.evaluation/'metrics.json').read_text());entries=report['episodes']
    pred=FrozenPredictor(ROOT/'artifacts/m21/model/best.pt');g=pred.schema['groups']
    noise=np.asarray(json.loads((ROOT/'artifacts/c2/probe_calibration/calibration.json').read_text())['scale'])
    output=[]
    for low in entries:
        if low['mu']!=.1:continue
        high=next(e for e in entries if e['K']==low['K'] and e['D']==low['D'] and e['mode']==low['mode'] and e['mu']==.8)
        paths=[a.evaluation/(e['episode_id']+'.npz') for e in (low,high)];data=[dict(np.load(p)) for p in paths]
        for stage in ('probe','late_sliding'):
            clock=max(low['trigger_s'] or 0,high['trigger_s'] or 0) if stage=='probe' else 10.5
            samples=[];pred_own=[];intervention=None
            for d,e in zip(data,(low,high)):
                t=int(np.argmin(abs(d['time']-clock)));n=min(20,len(d['state'])-t-1)
                if n<20 or t<8:break
                h,past=get_history(d,t);actions=d['action'][t:t+n];base=d['state'][:8,-12:].mean(0)
                samples.append(summary(h[-1],d['state'][t+1:t+n+1],g,base))
                means,_=pred.observe(h,past,actions,np.array([[e['K'],e['D'],e['mu']]]),QREF,base)
                pred_own.append(means[0])
                if e['mu']==.1:
                    means,_=pred.observe(h,past,actions,np.array([[e['K'],e['D'],.1],[e['K'],e['D'],.8]]),QREF,base)
                    intervention=means[1]-means[0]
            if len(samples)!=2:continue
            observed=samples[1]-samples[0];own=pred_own[1]-pred_own[0]
            row=dict(K=low['K'],D=low['D'],mode=low['mode'],stage=stage,window_s=1.,start_clock_s=clock,
                low_episode=low['episode_id'],high_episode=high['episode_id'],
                observed_high_minus_low=observed.tolist(),predicted_own_histories_high_minus_low=own.tolist(),
                same_low_history_actions_mu_intervention=intervention.tolist())
            for key,sl in [('q',slice(0,4)),('dq',slice(4,8)),('relative',slice(8,9)),('load',slice(9,21))]:
                row[key]=dict(observed_scale_rms=float(np.sqrt(np.mean((observed[sl]/noise[sl])**2))),
                    model_intervention_scale_rms=float(np.sqrt(np.mean((intervention[sl]/noise[sl])**2))),
                    predicted_own_history_scale_rms=float(np.sqrt(np.mean((own[sl]/noise[sl])**2))))
            output.append(row)
    dump(a.evaluation/'mu_diagnosis.json',dict(pairs=output,
        note='Observed paired effects include different histories and possible GPU/FEM variability. Same-history model intervention isolates model sensitivity only. No repeated-seed physical distinguishability claim.'))
    print('C2_MU_DIAGNOSIS_COMPLETE',len(output))
if __name__=='__main__':main()
