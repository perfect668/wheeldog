"""Frozen M2.1 model adapter; rejects unsupported/missing model history explicitly."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from dynadress.models.structured_world_model import load_structured_checkpoint
from dynadress.inference.belief import conditions
from dynadress.inference.segments import summary

CHECKPOINT_SHA256 = '55b895974648a64a98910db5291a87363fe07b985c74ab70cad1ea6c42a89525'


class FrozenPredictor:
    def __init__(self, checkpoint, device='cpu'):
        if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('C2 requires the unmodified M2.1 checkpoint')
        self.model, _ = load_structured_checkpoint(checkpoint, device)
        self.device = device; self.schema = self.model.schema
        for p in self.model.parameters(): p.requires_grad_(False)

    def rollout(self, history, past_actions, actions, grid, q_ref, prior=False):
        n = len(grid)
        if not all(np.isfinite(x).all() for x in (history, past_actions, actions)):
            raise ValueError('Missing model history/action: skip inference, never zero-impute')
        def tensor(x): return torch.as_tensor(np.asarray(x).copy(), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            function=self.model.prior_rollout if prior else self.model.rollout
            result=function(tensor(history)[None].expand(n,-1,-1),
                tensor(past_actions)[None].expand(n,-1,-1), tensor(actions)[None].expand(n,-1,-1),
                tensor(conditions(grid,q_ref))).cpu().numpy()
            if not np.isfinite(result).all():raise FloatingPointError('Nonfinite frozen-model prediction')
            return result

    def observe(self, history, past_actions, actions, grid, q_ref, baseline):
        pred = self.rollout(history,past_actions,actions,grid,q_ref)
        return summary(history[-1],pred,self.schema['groups'],baseline), pred
