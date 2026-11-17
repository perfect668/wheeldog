"""History-conditioned residual dynamics with a constant-velocity physics prior.

Known properties enter a distinct conditioning port. This M2 baseline does not
identify latent attributes, infer contact forces, or plan actions.
"""
import torch
from torch import nn


class CoupledWorldModel(nn.Module):
    def __init__(self, normalizer, feature_schema, hidden=96, transport_prior='action_translation'):
        super().__init__()
        self.schema=feature_schema;self.hidden=hidden;self.transport_prior=transport_prior
        for name,value in normalizer.items():self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        ns=len(normalizer['state_mean']);na=len(normalizer['action_mean']);nc=len(normalizer['condition_mean'])
        self.history=nn.GRU(ns+na,hidden,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden+na+nc,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,ns))
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.head[-1].weight,std=.001)

    def prior(self,current,previous=None,action=None):
        next_state=current.clone();g=self.schema['groups'];dt=self.schema['dt_s']
        for pos,vel in [('human_q','human_dq'),('robot_q','robot_dq')]:
            ps=slice(*g[pos]);vs=slice(*g[vel]);next_state[...,ps]=current[...,ps]+dt*current[...,vs]
        if self.transport_prior=='action_translation':
            # Native vertex attachments transmit commanded translation to the
            # cuff; learned residuals represent deformation and human motion.
            delta=action[...,18:24].reshape(*action.shape[:-1],2,3).mean(-2)
            sl=slice(*g['cuff_center']);next_state[...,sl]=current[...,sl]+delta
            sl=slice(*g['cuff_points']);next_state[...,sl]=current[...,sl]+delta.repeat(*([1]*(delta.ndim-1)),8)
            normal=current[...,slice(*g['cuff_normal'])]
            normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(.1)
            sl=slice(*g['progress']);next_state[...,sl]=current[...,sl]+(delta*normal).sum(-1,keepdim=True)
        elif previous is not None and self.transport_prior:
            # Short-horizon geometric transport prior. Load, shape, velocities
            # and rotations retain their last observed value before residuals.
            for name in ['cuff_center','cuff_points','progress']:
                sl=slice(*g[name]);next_state[...,sl]=2*current[...,sl]-previous[...,sl]
        return next_state

    def forward(self, states, previous_actions, candidate_action, attributes):
        # states[...,j,:] is o_j; previous_actions[...,j,:] is a_{j-1}.
        x=(states-self.state_mean)/self.state_std
        past=(previous_actions-self.action_mean)/self.action_std
        _,h=self.history(torch.cat([x,past],-1))
        action=(candidate_action-self.action_mean)/self.action_std
        condition=(attributes-self.condition_mean)/self.condition_std
        residual=self.head(torch.cat([h[-1],action,condition],-1))
        return self.prior(states[:,-1],states[:,-2],candidate_action)+residual*self.state_std

    def rollout(self, states, previous_actions, candidate_actions, attributes):
        """Open loop: after initial history, no measured future state is read."""
        predictions=[]
        for action in candidate_actions.unbind(1):
            y=self(states,previous_actions,action,attributes);predictions.append(y)
            states=torch.cat([states[:,1:],y[:,None]],1)
            previous_actions=torch.cat([previous_actions[:,1:],action[:,None]],1)
        return torch.stack(predictions,1)
