"""Streaming episode log with explicit signal provenance and unavailable measurements."""
import json
import time
from pathlib import Path
import numpy as np

GROUPS=('solver_state','solver_contact','estimated_load','commanded_wrench','measured_wrench')

def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)

class Episode:
    def __init__(self,path,metadata):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True)
        self.meta=dict(metadata,complete=False,outcome=None)
        (self.path/'metadata.json').write_text(json.dumps(self.meta,indent=2,default=serial,allow_nan=False))
        self.file=(self.path/'samples.jsonl').open('w');self.last=-1
    def write(self,physics_step,control_step,simulation_s,*,action=None,solver_state=None,solver_contact=None,estimated_load=None,commanded_wrench=None,measured_wrench=None,privileged=None):
        if physics_step<=self.last:raise ValueError('Non-monotonic physical step')
        row={'physics_step':physics_step,'control_step':control_step,'simulation_time_s':simulation_s,'sample_wall_time_ns':time.time_ns(),'action':action,'privileged':privileged}
        for key,value in zip(GROUPS,(solver_state,solver_contact,estimated_load,commanded_wrench,measured_wrench)):
            row[key]={'available':value is not None,'value':value}
        payload=json.dumps(row,default=serial,allow_nan=False);self.file.write(payload+'\n');self.file.flush();self.last=physics_step
    def close(self,outcome,reason):
        self.file.close();self.meta.update(complete=True,outcome=outcome,termination_reason=reason)
        (self.path/'metadata.json').write_text(json.dumps(self.meta,indent=2,default=serial,allow_nan=False))
