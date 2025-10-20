import argparse, torch, numpy as np
from .hparams import Cfg
from .models.resnet3d import resnet3d_18
from .models.simple3dcnn import Simple3DCNN

@torch.no_grad()
def main(args):
    cfg = Cfg(args.config).cfg
    x = np.load(args.input).astype(np.float32)  # HxWxC
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    H, W, C = x.shape
    winH, winW = cfg['data']['window']
    y0 = max(0, (H - winH)//2); x0 = max(0, (W - winW)//2)
    patch = x[y0:y0+winH, x0:x0+winW, :]
    if patch.shape[0] < winH or patch.shape[1] < winW:
        padH = winH - patch.shape[0]; padW = winW - patch.shape[1]
        patch = np.pad(patch, ((0,padH),(0,padW),(0,0)), mode='reflect')
    patch = np.transpose(patch, (2,0,1))
    patch = np.expand_dims(patch, 0)  # (1, C, H, W)

    m = resnet3d_18(in_channels=1, base=cfg['model']['width'], num_classes=len(cfg['classes']), dropout=cfg['model']['dropout']) if cfg['model']['name']=='resnet3d_18' else Simple3DCNN(1, cfg['model']['width'], len(cfg['classes']))
    state = torch.load(args.ckpt, map_location='cpu')
    key = 'state_dict' if 'state_dict' in state else None
    m.load_state_dict(state[key] if key else state, strict=False)
    m.eval()

    logits = m(torch.from_numpy(patch).float())
    prob = torch.softmax(logits, dim=1).numpy()[0]
    print({cls: float(p) for cls, p in zip(cfg['classes'], prob)})

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--input', type=str, required=True)
    args = p.parse_args()
    main(args)
