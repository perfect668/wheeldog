"""Score paired native C2 episodes; truth is accessed only in this offline tool.

Causal prediction uses the last arrived online belief at each anchor, never a
posterior that has assimilated a future evaluation target.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.inference.belief import JointBelief,hypotheses,entropy
from dynadress.inference.predictor import FrozenPredictor,CHECKPOINT_SHA256
from dynadress.human.geometry import axial_progress,arm_landmarks
from scripts.c2_replay import QREF,dump,get_history
from scripts.c2_metrics import error_components


def loadlines(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def belief_at(updates,t,mode='full'):
    arrived=[x for x in updates if x['observed_until_s']<=t+1e-6]
    return arrived[-1]['after'][mode] if arrived else JointBelief().summary()


def score_belief(s,truth):
    b=JointBelief();w=np.asarray(s['joint']);kd,mu=b.marginal(w)
    ks=np.unique(b.grid[:,0]);ds=np.unique(b.grid[:,1])
    pk=np.asarray([w[b.grid[:,0]==v].sum() for v in ks]);pd=np.asarray([w[b.grid[:,1]==v].sum() for v in ds])
    true=np.isclose(b.grid,truth,atol=1e-6).all(-1)
    classes=s['effective_sliding_resistance'];label='low' if truth[2]<=.2 else 'high' if truth[2]>=.5 else 'medium'
    return dict(p_true_joint=float(w[true].sum()),p_true_KD=float(kd[np.isclose(b.kd,truth[:2]).all(-1)].sum()),
        p_true_K=float(pk[np.isclose(ks,truth[0])].sum()),p_true_D=float(pd[np.isclose(ds,truth[1])].sum()),
        p_true_mu=float(mu[np.isclose(b.mu,truth[2])].sum()),p_true_resistance_class=classes[label],
        true_resistance_class=label,predicted_resistance_class=max(classes,key=classes.get),
        K_abs_error=abs(float(s['mean'][0])-truth[0]),D_abs_error=abs(float(s['mean'][1])-truth[1]),
        K_map_correct=bool(np.isclose(ks[pk.argmax()],truth[0])),D_map_correct=bool(np.isclose(ds[pd.argmax()],truth[1])),
        joint_map_K_correct=bool(np.isclose(s['map'][0],truth[0])),joint_map_D_correct=bool(np.isclose(s['map'][1],truth[1])),
        resistance_class_correct=max(classes,key=classes.get)==label,entropy_KD=entropy(kd),entropy_mu=entropy(mu),
        p_KD=kd.tolist(),p_K=pk.tolist(),p_D=pd.tolist(),p_mu=mu.tolist())


def prediction_metrics(d,updates,predictor):
    errors={};geometry={};grid=hypotheses();truth=[d['condition'][0],d['condition'][4],d['condition'][8]]
    expanded=np.vstack([grid,[12,1.25,.3],truth]);g=predictor.schema['groups']
    for t in range(8,len(d['state'])-10,5):
        h,p=get_history(d,t);actions=d['action'][t:t+10]
        pred=predictor.rollout(h,p,actions,expanded,QREF)
        yp={'middle':pred[-2],'oracle':pred[-1]}
        for mode in ('human_only','full'):
            w=np.asarray(belief_at(updates,float(d['time'][t]),mode)['joint'])
            yp[mode]=np.einsum('n,ntd->td',w,pred[:len(grid)])
        prior=predictor.rollout(h,p,actions,np.asarray([truth]),QREF,prior=True)[0]
        for n in (5,10):
            true=d['state'][t+n];target=true[g['progress'][0]]
            for mode,value in yp.items():
                diff=value[n-1]-true
                for name,values in error_components(diff,g).items():
                    errors.setdefault(f'{mode}/{n*.05:.2f}/{name}',[]).extend(values.tolist())
            for mode,value in [('oracle_model',yp['oracle']),('posterior_model',yp['full']),('impedance_translation_prior',prior)]:
                fk=axial_progress(value[n-1,:4],value[n-1,slice(*g['cuff_center'])])
                geometry.setdefault(f'{mode}/{n*.05:.2f}/fk_progress_m',[]).append(float(fk-target))
                geometry.setdefault(f'{mode}/{n*.05:.2f}/native_progress_m',[]).append(float(value[n-1,g['progress'][0]]-target))
    return {k:float(np.sqrt(np.mean(np.asarray(v)**2))) for k,v in errors.items()}, {k:float(np.sqrt(np.mean(np.asarray(v)**2))) for k,v in geometry.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,default=ROOT/'dynadress/configs/c2_paired/plan.json')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/c2/evaluation');p.add_argument('--no-predictions',action='store_true')
    a=p.parse_args();import torch;torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text());entries=plan['episodes'];a.output.mkdir(parents=True,exist_ok=True)
    predictor=FrozenPredictor(ROOT/'artifacts/m21/model/best.pt') if not a.no_predictions else None
    loaded={}
    for e in entries:
        cfg=json.loads((ROOT/e['config']).read_text());path=ROOT/'artifacts'/cfg['output_group']/e['episode_id']
        if not (path/'c2_result.json').exists():raise RuntimeError(f'Incomplete episode: {path}')
        seq=dict(np.load(path/'observable_sequence.npz'));rows=loadlines(path/'samples.jsonl')
        updates=loadlines(path/'belief.jsonl');result=json.loads((path/'result.json').read_text())
        selection=json.loads((path/'probe_selection.json').read_text()) if (path/'probe_selection.json').exists() else None
        loaded[e['episode_id']]=(cfg,path,seq,rows,updates,result,selection)
    metrics=[];curves={};consistency=[];landmark_error=[]
    for e in entries:
        cfg,path,d,rows,updates,result,selection=loaded[e['episode_id']]
        peers=[loaded[x['episode_id']] for x in entries if x['pair_id']==e['pair_id']]
        triggers=[x[-1]['trigger_s'] for x in peers if x[-1]]
        pre=min(triggers) if triggers else 0.;post=max(triggers)+1.0 if triggers else 1.0
        truth=np.array([e['K'],e['D'],e['mu']]);times=d['time']
        # Native controller time is relative to world construction; stored feature
        # time is the same relative time. Raw samples carry absolute world time.
        before=score_belief(belief_at(updates,pre),truth);after=score_belief(belief_at(updates,post),truth)
        final=score_belief(json.loads((path/'c2_result.json').read_text())['final']['full'],truth)
        load=d['state'][:,-12:].reshape(-1,2,6);force=np.linalg.norm(load[...,:3],axis=-1)
        sel=(times>=pre)&(times<=post+1e-6);duration=np.diff(times)
        total=float(np.sum((force[1:].sum(-1)+force[:-1].sum(-1))*.5*duration))
        window_duration=np.maximum(0.,np.minimum(times[1:],post)-np.maximum(times[:-1],pre))
        window_integral=float(np.sum((force[1:].sum(-1)+force[:-1].sum(-1))*.5*window_duration))
        aligned=np.flatnonzero(times<=post+1e-6);ix=aligned[-1] if len(aligned) else 0
        g=json.loads((ROOT/'artifacts/m2/dataset/features.json').read_text())['groups']
        geom=axial_progress(d['state'][:,:4],d['state'][:,slice(*g['cuff_center'])]);diff=geom-d['state'][:,g['progress'][0]]
        consistency.extend(diff.tolist())
        valid=[r for r in rows if r['solver_state']['value']['geometry'] is not None]
        for r in valid:
            s=r['solver_state']['value'];fk=arm_landmarks(s['human_q_rad'])
            landmark_error.append(float(np.linalg.norm(fk['wrist']-s['human_wrist_m'])))
        # Validate complete feature/action replay against actual native sent log.
        from dynadress.models.features import extract
        paired=[(r,extract(r)) for r in valid];paired=[(r,x) for r,x in paired if x is not None]
        valid=[r for r,_ in paired];extracted=[x for _,x in paired]
        sraw=np.asarray([x[0] for x in extracted]);araw=np.asarray([x[1] for x in extracted])
        if sraw.shape!=d['state'].shape or not np.allclose(sraw,d['state'],atol=1e-6) or not np.allclose(araw,d['action'],atol=1e-6):
            raise AssertionError('Online feature/action mismatch with raw solver log')
        # Keep the native counters and timestamps, never reconstruct them from
        # nominal frequency. These are provenance, not model inference inputs.
        for field in ('physics_step','control_step','sample_wall_time_ns','simulation_time_s'):
            d[field]=np.asarray([r[field] for r in valid])
        d['task_stage']=np.asarray([r['action']['state'] for r in valid])
        d['condition']=np.r_[[e['K']]*4,[e['D']]*4,e['mu'],QREF].astype(np.float32)
        prediction,geometry=prediction_metrics(d,updates,predictor) if predictor else ({},{})
        actions=loadlines(path/'c2_actions.jsonl')
        during=force[sel]
        phases=[u for u in updates if pre<=u['observed_until_s']<=post+1e-6]
        row=dict(**e,success=bool(result['success']),termination_reason=result['termination_reason'],
            trigger_s=None if selection is None else selection['trigger_s'],selected=None if selection is None else selection['selected'],
            matched_pre_s=pre,matched_post_s=post,before=before,after=after,final=final,
            matched_post_observed=bool(times[-1]>=post-1e-6),
            human_only_after=score_belief(belief_at(updates,post,'human_only'),truth),
            relative_progress_at_post_m=float(d['state'][ix,g['progress'][0]]),
            final_progress_m=float(d['state'][-1,g['progress'][0]]),
            wrist_peak_L_R_N=force.max(0).tolist(),integrated_external_force_norm_sum_Ns=total,
            matched_window_integrated_external_force_norm_sum_Ns=window_integral,
            targets_reached=int(result.get('hand_entered',False))+int(result.get('wrist_entered',False))+int(result['success']),
            probe_window_peak_L_R_N=during.max(0).tolist() if len(during) else None,
            tracking_interrupted_samples=sum(x['tracking_interrupted'] for x in actions),
            effective_contact_summary_duration_s=sum(max(0.,min(x['observed_until_s'],post)-max(x['segment_start_s'],pre)) for x in phases if x['phase']!='precontact'),
            probe_opportunity_duration_s=1.0 if selection else 0.,control_step_actions=len(actions),
            simulation_duration_s=result['simulation_duration_s'],prediction_rmse=prediction,geometry_prediction_rmse=geometry,
            source_hashes={f:hashlib.sha256((path/f).read_bytes()).hexdigest() for f in ('samples.jsonl','observable_sequence.npz','belief.jsonl','result.json')})
        metrics.append(row);np.savez_compressed(a.output/(e['episode_id']+'.npz'),**{k:v for k,v in d.items() if k!='condition'})
        # Truth remains in metrics, not the released inference-input sequence.
        dump(a.output/(e['episode_id']+'_beliefs.json'),dict(hypotheses=hypotheses().tolist(),updates=updates,selection=selection))
        (a.output/(e['episode_id']+'_actions.jsonl')).write_text((path/'c2_actions.jsonl').read_text())
        curves[e['episode_id']]=dict(times=times,states=d['state'],updates=updates)
        print('C2_EVALUATED',e['episode_id'],result['success'],flush=True)
    aggregate={}
    for mode in ('passive','fixed','active'):
        rr=[r for r in metrics if r['mode']==mode]
        aggregate[mode]=dict(episodes=len(rr),successes=sum(r['success'] for r in rr),
            selected={s:sum(r['selected']==s for r in rr) for s in ('retreat_readvance','pause_advance','roll_differential','continue')},
            **{key:float(np.mean([r['after'][key] for r in rr])) for key in ('p_true_joint','p_true_KD','p_true_resistance_class','K_abs_error','D_abs_error','resistance_class_correct')},
            final_K_map_accuracy=float(np.mean([r['final']['K_map_correct'] for r in rr])),
            final_D_map_accuracy=float(np.mean([r['final']['D_map_correct'] for r in rr])))
    report=dict(episodes=metrics,aggregate=aggregate,checkpoint_sha256=CHECKPOINT_SHA256,
        plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
        fk_native_progress_consistency_rmse_m=float(np.sqrt(np.mean(np.asarray(consistency)**2))),
        fk_native_wrist_max_error_m=float(np.max(landmark_error)),
        load_note='20Hz sampled net external wrist estimate. Integral of sum of force norms; not signed impulse or garment-only force. Peaks are not 240Hz peaks.',
        comparison_note='Post times use latest paired trigger+1.0s; only beliefs already arrived by that time are scored. Failures retained.')
    dump(a.output/'metrics.json',report)
    flat=[]
    for r in metrics:
        row={k:r[k] for k in ('episode_id','pair_id','mode','K','D','mu','success','termination_reason','selected','trigger_s','matched_pre_s','matched_post_s','relative_progress_at_post_m','final_progress_m','integrated_external_force_norm_sum_Ns','matched_window_integrated_external_force_norm_sum_Ns','targets_reached','simulation_duration_s','tracking_interrupted_samples','probe_opportunity_duration_s')}
        for stage in ('before','after','final'):
            row.update({f'{stage}_{k}':v for k,v in r[stage].items() if isinstance(v,(float,int,bool,str))})
        row['wrist_peak_L_N'],row['wrist_peak_R_N']=r['wrist_peak_L_R_N'];flat.append(row)
    with (a.output/'episode_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    print('C2_EVALUATION_COMPLETE',json.dumps(aggregate),flush=True)
if __name__=='__main__':main()
