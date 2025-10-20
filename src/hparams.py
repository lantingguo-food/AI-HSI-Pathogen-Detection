import yaml
from .utils.seed import set_seed

class Cfg:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        set_seed(self.cfg.get('seed', 42))
    def __getitem__(self, k):
        return self.cfg[k]
    def get(self, k, default=None):
        return self.cfg.get(k, default)
