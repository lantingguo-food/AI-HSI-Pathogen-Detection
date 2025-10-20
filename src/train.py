import os, argparse, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .hparams import Cfg
from .utils.logging_utils import ensure_dir, console
from .utils.metrics import classification_metrics
from .data.hsi_dataset import HSICubeDataset
from .models.resnet3d import resnet3d_18
from .models.simple3dcnn import Simple3DCNN

MODEL_ZOO = {
    'resnet3d_18': resnet3d_18,
    'simple3d': lambda in_channels, base, num_classes, dropout: Simple3DCNN(in_channels, base, num_classes)
}

def build_model(cfg, bands, n_classes):
    name = cfg['model']['name']
    base = cfg['model'].get('width', 32)
    dropout = cfg['model'].get('dropout', 0.1)
    if name == 'resnet3d_18':
        m = resnet3d_18(in_channels=1, base=base, num_classes=n_classes, dropout=dropout)
    elif name == 'simple3d':
        m = Simple3DCNN(in_channels=1, base=base, num_classes=n_classes)
    else:
        raise ValueError('Unknown model')
    return m

def load_pretrained(m, ckpt_path, freeze_layers):
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        key = 'state_dict' if 'state_dict' in state else None
        m.load_state_dict(state[key] if key else state, strict=False)
        console.print(f"[green]Loaded pretrained from {ckpt_path}[/green]")
    for name, module in m.named_children():
        if name in freeze_layers:
            for p in module.parameters():
                p.requires_grad = False
            console.print(f"[yellow]Froze layer: {name}[/yellow]")
    return m

def train_epoch(model, loader, optimizer, scaler, device, criterion):
    model.train()
    losses = []
    all_probs, all_labels = [], []
    for batch in loader:
        x = batch['x'].to(device, non_blocking=True).float()
        y = batch['y'].to(device, non_blocking=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x); loss = criterion(logits, y)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        losses.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        all_probs.append(probs); all_labels.append(y.cpu().numpy())
    import numpy as np
    all_probs = np.concatenate(all_probs); all_labels = np.concatenate(all_labels)
    mets = classification_metrics(all_labels, all_probs)
    return float(np.mean(losses)), mets

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    losses = []; all_probs, all_labels = [], []
    for batch in loader:
        x = batch['x'].to(device).float(); y = batch['y'].to(device)
        logits = model(x); loss = criterion(logits, y)
        losses.append(loss.item())
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        all_probs.append(probs); all_labels.append(y.cpu().numpy())
    import numpy as np
    all_probs = np.concatenate(all_probs); all_labels = np.concatenate(all_labels)
    mets = classification_metrics(all_labels, all_probs)
    return float(np.mean(losses)), mets

def main(args):
    cfg = Cfg(args.config).cfg
    ensure_dir(cfg['paths']['run_dir'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = HSICubeDataset(cfg['paths']['train_dir'], cfg, split='train')
    val_ds = HSICubeDataset(cfg['paths']['val_dir'], cfg, split='val')

    bands = train_ds.cubes[0].shape[-1]
    n_classes = len(cfg['classes'])

    train_loader = DataLoader(train_ds, batch_size=cfg['trainer']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['trainer']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True)

    model = build_model(cfg, bands, n_classes)
    model = load_pretrained(model, cfg['model'].get('pretrained_ckpt', None), cfg['model'].get('freeze_layers', []))
    model.to(device)

    if cfg['trainer']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['trainer']['lr'], weight_decay=cfg['trainer']['weight_decay'])
    elif cfg['trainer']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['trainer']['lr'])
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['trainer']['lr'], momentum=0.9, weight_decay=cfg['trainer']['weight_decay'])

    if cfg['trainer']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['trainer']['max_epochs'])
    elif cfg['trainer']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    class_weights = cfg['trainer'].get('class_weights', None)
    if class_weights:
        cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
    else:
        cw = None
    criterion = nn.CrossEntropyLoss(weight=cw)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['trainer'].get('amp', True)) and device.type=='cuda')

    best_f1, best_path = -1.0, None
    patience = cfg['trainer'].get('patience', 15)
    wait = 0
    for epoch in range(1, cfg['trainer']['max_epochs']+1):
        tr_loss, tr_m = train_epoch(model, train_loader, optimizer, scaler, device, criterion)
        va_loss, va_m = eval_epoch(model, val_loader, device, criterion)
        if scheduler: scheduler.step()
        console.print(f"[epoch {epoch}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_f1={va_m['f1']:.4f} val_auc={va_m['auc']:.4f}")
        if va_m['f1'] > best_f1:
            best_f1 = va_m['f1']
            ensure_dir(cfg['paths']['run_dir'])
            best_path = os.path.join(cfg['paths']['run_dir'], 'best.pt')
            torch.save(model.state_dict(), best_path)
            console.print(f"[green]Saved best to {best_path}[/green]")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                console.print("[yellow]Early stopping[/yellow]")
                break
    console.print(f"Best F1: {best_f1:.4f}; ckpt: {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args)
