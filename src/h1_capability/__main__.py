from .experiment import run
import argparse
p=argparse.ArgumentParser(description='M1-C offline force and sampled geometric capability')
p.add_argument('--config',default='configs/m1_capability.yaml');p.add_argument('--output',required=True)
p.add_argument('--stage',choices=['force','reach','sensitivity'],required=True);p.add_argument('--workers',type=int,default=2)
a=p.parse_args();run(a.config,a.output,a.stage,a.workers)
