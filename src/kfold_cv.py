import argparse, os
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
from .hparams import Cfg
from .utils.logging_utils import console
import subprocess

def main(args):
    cfg = Cfg(args.config).cfg
    data_dir = cfg['paths']['train_dir']
    with open(os.path.join(data_dir, cfg['data']['labels_file']), 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    X = np.arange(len(lines))
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=cfg['seed'])

    tmp = Path('runs/kfold_tmp'); tmp.mkdir(parents=True, exist_ok=True)
    for i, (tr, va) in enumerate(kf.split(X), 1):
        tr_file = tmp / f'train_fold{i}.csv'
        va_file = tmp / f'val_fold{i}.csv'
        with open(tr_file, 'w') as f:
            for idx in tr: f.write(lines[idx] + '\n')
        with open(va_file, 'w') as f:
            for idx in va: f.write(lines[idx] + '\n')
        console.print(f"[bold]Fold {i}[/bold]: train={len(tr)} val={len(va)}")
        cmd = ['python', '-m', 'src.train', '--config', args.config]
        env = os.environ.copy()
        env['OVERRIDE_TRAIN_LABELS'] = str(tr_file)
        env['OVERRIDE_VAL_LABELS'] = str(va_file)
        subprocess.run(cmd, env=env, check=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--folds', type=int, default=5)
    args = p.parse_args()
    main(args)
