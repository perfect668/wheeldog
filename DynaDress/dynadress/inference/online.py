"""Causal C2 observer and selector: only observable feature vectors enter here.

No parameter config, episode identifier, old probe profile, solver contact truth,
privileged hold torque or outcome is an input. q_ref is common and known.
"""
import json
from pathlib import Path
import numpy as np
from dynadress.inference.belief import JointBelief, SegmentLikelihood, expected_information_gain
from dynadress.inference.predictor import FrozenPredictor
from dynadress.inference.segments import summary, phase
from dynadress.inference.probing import primitive,candidate_actions,proposal_constraints,DURATION,CANDIDATES


class OnlineIdentifier:
    def __init__(self, root, output, mode, q_ref, seed=20261119):
        if mode not in ('passive','fixed','active'):raise ValueError(mode)
        self.mode=mode;self.q_ref=np.asarray(q_ref);self.output=Path(output)
        self.predictor=FrozenPredictor(root/'artifacts/m21/model/best.pt','cpu')
        self.groups=self.predictor.schema['groups']
        self.likelihood=SegmentLikelihood(json.loads((root/'artifacts/c2/calibration/calibration.json').read_text())['scale'])
        self.probe_likelihood=SegmentLikelihood(json.loads((root/'artifacts/c2/probe_calibration/calibration.json').read_text())['scale'])
        self.beliefs={name:JointBelief() for name in ('human_only','full')}
        self.states=[];self.actions=[];self.times=[];self.baseline=None;self.next_anchor=8
        self.phase='precontact';self.last_update=None;self.selected=None;self.trigger=None
        self.stable_samples=0;self.rng=np.random.default_rng(seed);self.updates=0
        self.belief_file=(self.output/'belief.jsonl').open('w');self.action_file=(self.output/'c2_actions.jsonl').open('w')
        self.selection=None;self.scores=None

    def observe(self,state,t):
        if state is None:
            if self.states:raise ValueError('Observable history became unavailable; explicit inference failure')
            return
        self.states.append(np.asarray(state));self.times.append(float(t))
        n=len(self.states)-1
        if self.baseline is None and len(self.states)>=8:self.baseline=np.mean(np.asarray(self.states[:8])[:,-12:],axis=0)
        if n < self.next_anchor+5:return
        t0=self.next_anchor;h=np.asarray(self.states[t0-7:t0+1]);past=np.asarray(self.actions[t0-8:t0]);a=np.asarray(self.actions[t0:t0+5])
        if len(past)!=8 or len(a)!=5:raise ValueError('Action/state timestamp alignment failed')
        future=np.asarray(self.states[t0+1:t0+6]);self.phase=phase(h[-1],future,self.groups,self.q_ref,self.baseline)
        pred,_=self.predictor.observe(h,past,a,self.beliefs['full'].grid,self.q_ref,self.baseline)
        y=summary(h[-1],future,self.groups,self.baseline)
        before={k:b.summary() for k,b in self.beliefs.items()}
        for mode,b in self.beliefs.items():b.update(y,pred,self.likelihood,self.phase,mode=='human_only')
        self.last_update=dict(segment_start_s=self.times[t0],observed_until_s=self.times[n],phase=self.phase,
            anchor_index=t0,observation=y.tolist(),before=before,after={k:b.summary() for k,b in self.beliefs.items()})
        self.belief_file.write(json.dumps(self.last_update)+'\n');self.belief_file.flush()
        self.next_anchor+=5;self.updates+=1

    def control(self,t,stage,tracking,commanded,haxis,lateral,speed,ik_feasible=None):
        stable=max(tracking.values())<.006
        self.stable_samples=self.stable_samples+1 if stable else 0
        if self.trigger is None and stage in ('insert','advance') and self.phase=='sliding' and self.stable_samples>=5:
            self.trigger=float(t);self.selected='continue'
            h=np.asarray(self.states[-8:]);past=np.asarray(self.actions[-8:])
            scores={};proposals={}
            # Common random numbers compare candidate responses under the same
            # hypothesis/noise samples, reducing Monte Carlo ranking variance.
            comparison_seed=int(self.rng.integers(2**31))
            for candidate in CANDIDATES:
                a=candidate_actions(candidate,commanded,haxis,lateral,speed)
                limits=proposal_constraints(a);proposals[candidate]=a
                limits['ik_feasible']=True if ik_feasible is None else bool(ik_feasible(a))
                limits['feasible']=limits['feasible'] and limits['ik_feasible']
                pred,_=self.predictor.observe(h,past,a,self.beliefs['full'].grid,self.q_ref,self.baseline)
                score=expected_information_gain(self.beliefs['full'],pred,self.probe_likelihood,np.random.default_rng(comparison_seed),phase='sliding')
                scores[candidate]=dict(**score,**limits)
            if self.mode=='fixed':self.selected='retreat_readvance'
            elif self.mode=='active':
                valid=[name for name in CANDIDATES if scores[name]['feasible']]
                if valid:
                    best=max(valid,key=lambda name:scores[name]['score'])
                    if scores[best]['score']-scores['continue']['score']>.01:self.selected=best
            self.scores=scores
            self.selection=dict(trigger_s=float(t),stage=stage,selected=self.selected,mode=self.mode,duration_s=DURATION,
                belief_before={k:b.summary() for k,b in self.beliefs.items()},scores=scores,
                proposal_note='constant observed axis; actual feedback servo commands are recorded separately')
            (self.output/'probe_selection.json').write_text(json.dumps(self.selection,indent=2))
            np.savez_compressed(self.output/'probe_proposals.npz',**proposals)
        if self.trigger is None:return primitive('continue',0.)
        return primitive(self.selected,t-self.trigger)

    def sent(self,action,t,tracking):
        if not self.states:return
        if len(self.actions)!=len(self.states)-1:raise ValueError('One sent action per observed state required')
        self.actions.append(np.asarray(action))
        if self.trigger is not None and self.trigger<=t<=self.trigger+DURATION:
            self.action_file.write(json.dumps(dict(time_s=float(t),action=np.asarray(action).tolist(),
                tracking_m=tracking,tracking_interrupted=max(tracking.values())>=.006))+'\n');self.action_file.flush()

    def close(self):
        self.belief_file.close();self.action_file.close()
        if self.states:
            np.savez_compressed(self.output/'observable_sequence.npz',state=np.asarray(self.states),action=np.asarray(self.actions),time=np.asarray(self.times))
        (self.output/'c2_result.json').write_text(json.dumps(dict(mode=self.mode,trigger_s=self.trigger,selected=self.selected,
            updates=self.updates,final={k:b.summary() for k,b in self.beliefs.items()},hypotheses=self.beliefs['full'].grid.tolist()),indent=2))
