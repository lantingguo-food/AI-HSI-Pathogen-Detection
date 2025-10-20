from pathlib import Path
from rich.console import Console

console = Console()

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
