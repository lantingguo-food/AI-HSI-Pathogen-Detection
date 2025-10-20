import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from .hparams import Cfg
from .data.hsi_dataset import HSICubeDataset
from .models.resnet3d import resnet3d_18
from .models.simple3dcnn import Simple3DCNN
from .utils.metrics import classification_metrics
from .utils.logging_utils import console

@torch.no_grad()
def main(args):
    cfg = Cfg(args.config).cfg
    ds = HSICubeDataset(cfg['paths']['test_dir'], cfg, split='test')
    loader = DataLoader(ds, batch_size=cfg['trainer']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    n_classes = len(cfg['classes'])
    m = resnet3d_18(in_channels=1, base=cfg['model']['width'], num_classes=n_classes, dropout=cfg['model']['dropout']) if cfg['model']['name']=='resnet3d_18' else Simple3DCNN(1, cfg['model']['width'], n_classes)
    state = torch.load(args.ckpt, map_location='cpu')
    key = 'state_dict' if 'state_dict' in state else None
    m.load_state_dict(state[key] if key else state, strict=False)
    m.eval()

    all_probs, all_labels = [], []
    for b in loader:
        logits = m(b['x'].float())
        probs = torch.softmax(logits, dim=1)[:,1].numpy()
        all_probs.append(probs)
        all_labels.append(b['y'].numpy())
    y_prob = np.concatenate(all_probs); y_true = np.concatenate(all_labels)
    mets = classification_metrics(y_true, y_prob)
    console.print(mets)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--ckpt', type=str, required=True)
    args = p.parse_args()
    main(args)
