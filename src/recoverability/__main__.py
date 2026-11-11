"""One entry point for running and plotting EXP0."""
import argparse
from pathlib import Path
import yaml
from .experiment import run
from .plots import plot_all


def main():
    p=argparse.ArgumentParser(description='EXP0 toy simulation; CPU only')
    p.add_argument('--config',type=Path,default=Path('configs/exp0.yaml'))
    p.add_argument('--output',type=Path,default=Path('results/local'))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--mode',choices=['sweep','single','plots'],default='sweep')
    args=p.parse_args()
    if args.mode!='plots':
        if args.output.exists() and any(args.output.iterdir()):
            p.error('output directory is not empty; use a new directory to preserve raw evidence')
        config=yaml.safe_load(args.config.read_text())
        run(config,args.output,args.seed,args.mode)
    plot_all(args.output)


if __name__=='__main__':
    main()
