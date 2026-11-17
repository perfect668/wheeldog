"""Train only on whole training episodes; select on disjoint validation groups."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynadress.models.world_model import CoupledWorldModel


class Windows(Dataset):
    def __init__(self,folder,index,split,history=8,horizon=20,stride=1):
        self.episodes=[];self.windows=[];self.h=history;self.r=horizon
        for record in index:
            if record['split']!=split:continue
            d=dict(np.load(folder/record['file']));ei=len(self.episodes);self.episodes.append(d)
            # Each history frame includes its *previous* action. Index >= H.
            self.windows.extend((ei,t) for t in range(history,len(d['state'])-horizon,stride))
    def __len__(self):return len(self.windows)
    def __getitem__(self,i):
        ei,t=self.windows[i];d=self.episodes[ei];h=self.h;r=self.r
        return tuple(torch.from_numpy(v.copy()).float() for v in (
            d['state'][t-h+1:t+1],d['action'][t-h:t],d['action'][t:t+r],d['condition'],d['state'][t+1:t+r+1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=ROOT/'artifacts/m2/dataset');p.add_argument('--output',type=Path,default=ROOT/'artifacts/m2/model');p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--seed',type=int,default=1702);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device=='cuda':torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False
    a.output.mkdir(parents=True,exist_ok=True)
    index=json.loads((a.dataset/'index.json').read_text());fs=json.loads((a.dataset/'features.json').read_text())
    train=Windows(a.dataset,index,'train');val=Windows(a.dataset,index,'validation',stride=4)
    assert len(train)>0 and len(val)>0
    norm={}
    for key in ['state','action','condition']:
        values=np.concatenate([d[key] if key!='condition' else d[key][None] for d in train.episodes])
        norm[key+'_mean']=values.mean(0).tolist()
        norm[key+'_std']=np.maximum(values.std(0),fs[key+'_std_floor']).tolist()
    # All normalizers see training episodes only, including failed training runs.
    model=CoupledWorldModel(norm,fs).to(a.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0)
    validation=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
    weight=torch.ones(len(fs['state_names']),device=a.device)
    for group in ['robot_q','robot_dq','ee_pose']:weight[slice(*fs['groups'][group])]=.25
    for group in ['human_q','progress']:weight[slice(*fs['groups'][group])]=2.
    def loss(pred,target):return (((pred-target)/model.state_std).square()*weight).mean()
    config=dict(epochs=a.epochs,seed=a.seed,batch_size=a.batch_size,history=8,training_horizon=20,hidden=96,learning_rate=.001,transport_prior='action_translation',
        train_windows=len(train),validation_windows=len(val),normalization='training episodes only',
        split_sha256=hashlib.sha256((a.dataset/'split_groups.json').read_bytes()).hexdigest(),
        index_sha256=hashlib.sha256((a.dataset/'index.json').read_bytes()).hexdigest(),
        train_groups=sorted(set(r['parameter_group'] for r in index if r['split']=='train')),
        validation_groups=sorted(set(r['parameter_group'] for r in index if r['split']=='validation')),
        test_groups=sorted(set(r['parameter_group'] for r in index if r['split']=='test')),
        torch_version=torch.__version__,feature_schema=fs,normalizer=norm,
        loss='weighted normalized state MSE; .5 one-step + .5 open-loop twenty-step',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'dynadress/models/world_model.py',ROOT/'dynadress/models/features.py']})
    (a.output/'training_config.json').write_text(json.dumps(config,indent=2))
    best=float('inf');history=[];start=time.time()
    for epoch in range(a.epochs):
        model.train();total=0.;n=0
        for batch in loader:
            states,past,actions,cond,target=[v.to(a.device) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            pred=model.rollout(states,past,actions,cond)
            error=.5*loss(pred[:,0],target[:,0])+.5*loss(pred,target)
            error.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(error.detach())*len(states);n+=len(states)
        model.eval();vtotal=0.;vn=0
        with torch.no_grad():
            for batch in validation:
                states,past,actions,cond,target=[v.to(a.device) for v in batch]
                pred=model.rollout(states,past,actions,cond)
                err=.5*loss(pred[:,0],target[:,0])+.5*loss(pred,target)
                vtotal+=float(err)*len(states);vn+=len(states)
        metric=vtotal/vn;row=dict(epoch=epoch+1,train_loss=total/n,validation_loss=metric,elapsed_s=time.time()-start)
        history.append(row);(a.output/'learning_curve.json').write_text(json.dumps(history,indent=2))
        if metric<best:
            best=metric;torch.save(dict(state_dict=model.state_dict(),normalizer=norm,feature_schema=fs,hidden=96,transport_prior='action_translation',epoch=epoch+1,validation_loss=best),a.output/'best.pt')
        print(json.dumps(row),flush=True)
        if epoch>=30 and epoch+1-min(history,key=lambda r:r['validation_loss'])['epoch']>=20:break
    print('M2_TRAIN_COMPLETE',json.dumps(dict(epochs=len(history),best_validation_loss=best,wall_s=time.time()-start)),flush=True)


if __name__=='__main__':main()
