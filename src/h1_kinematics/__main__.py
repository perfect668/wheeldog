import argparse
from .experiment import run
p=argparse.ArgumentParser(description='M1 static URDF audit and kinematics only')
p.add_argument('--config',default='configs/m1_kinematics.yaml')
p.add_argument('--output',required=True)
p.add_argument('--seed',type=int)
a=p.parse_args()
run(a.config,a.output,a.seed)
