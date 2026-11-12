from .experiment import run
import argparse
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/m1_coupled.yaml');p.add_argument('--output',required=True)
p.add_argument('--stage',choices=['prepare','main','sensitivity','future'],required=True);p.add_argument('--workers',type=int,default=4)
a=p.parse_args();run(a.config,a.output,a.stage,a.workers)
