import argparse, torch, numpy as np
import torch.nn.functional as F
from ..hparams import Cfg
from ..models.resnet3d import resnet3d_18

class GradCAM3D:
    def __init__(self, model, target_layer_name='layer4'):
        self.model = model.eval()
        self.target_activations = None
        self.gradients = None
        target_layer = dict([*model.named_modules()])[target_layer_name]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self, module, inp, out):
        self.target_activations = out.detach()
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(torch.ones_like(score))
        weights = self.gradients.mean(dim=(2,3,4), keepdim=True)
        cam = F.relu((weights * self.target_activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode='trilinear', align_corners=False)
        cam = cam / (cam.max() + 1e-8)
        return cam

@torch.no_grad()
def main(args):
    cfg = Cfg(args.config).cfg
    x = np.load(args.input).astype(np.float32)  # HWC
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

    m = resnet3d_18(in_channels=1, base=cfg['model']['width'], num_classes=len(cfg['classes']), dropout=cfg['model']['dropout'])
    state = torch.load(args.ckpt, map_location='cpu')
    key = 'state_dict' if 'state_dict' in state else None
    m.load_state_dict(state[key] if key else state, strict=False)

    cam = GradCAM3D(m)
    x_t = torch.from_numpy(patch).float()
    heat = cam(x_t)
    np.save(args.output, heat.numpy())
    print(f"Saved CAM to {args.output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='config.yaml')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--input', type=str, required=True)
    p.add_argument('--output', type=str, default='cam.npy')
    args = p.parse_args()
    main(args)
