# Run in a fresh Kaggle notebook with Internet and GPU enabled.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import sys, subprocess, json, hashlib, platform, shutil, time, datetime

ROOT = Path('/kaggle/working/DA4022_FINAL_TODAY')
ROOT.mkdir(parents=True, exist_ok=True)
REPO = Path('/kaggle/working/capa')
DATA_CACHE = Path('/kaggle/working/da4022_downloads')
DATA_CACHE.mkdir(exist_ok=True)

# Restore the folder DA4022_FINAL_TODAY from a previous backup here BEFORE
# running subsequent cells if resuming a new session.
def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)

def journal(event, **details):
    with (ROOT / 'journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(time_utc=utcnow(), event=event, **details), default=str) + '\n')

def command(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with (ROOT / 'setup.log').open('a', encoding='utf-8') as f:
        f.write('\n' + repr(args) + '\n' + p.stdout)
    print(p.stdout[-2500:])
    p.check_returncode()

if not (REPO / '.git').exists():
    command(['git', 'clone', 'https://github.com/nv-dvl/capa.git', str(REPO)])

# Reuse the saved CAPA commit when resuming, rather than silently changing code.
old_env_path = ROOT / 'environment.json'
old_environment = json.loads(old_env_path.read_text()) if old_env_path.exists() else None
if old_environment and old_environment.get('capa_commit'):
    command(['git', '-C', str(REPO), 'checkout', old_environment['capa_commit']])

# Deliberately reuse Kaggle's torch/torchvision. No PEFT or TorchAO needed:
# the adaptation layers below are implemented directly in PyTorch.
command([sys.executable, '-m', 'pip', 'install', '-q',
         'huggingface_hub', 'einops', 'safetensors', 'pillow',
         'pandas', 'matplotlib', 'requests'])

import torch, numpy as np, pandas as pd
assert torch.cuda.is_available(), 'Enable GPU in Kaggle notebook settings.'
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

CAPA_COMMIT = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
import importlib.metadata
environment = dict(
    recorded_at=utcnow(), capa_commit=CAPA_COMMIT, python=platform.python_version(),
    packages={n: importlib.metadata.version(n) for n in
              ['torch', 'torchvision', 'numpy', 'huggingface_hub', 'einops', 'safetensors', 'pandas']},
    gpus=[dict(name=torch.cuda.get_device_name(i),
               GB=round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2))
          for i in range(torch.cuda.device_count())],
    note='Uses GPU 0 only. Two GPUs do not pool their VRAM. Determinism is best-effort; warnings are retained.')
if old_environment and old_environment['packages'] != environment['packages']:
    write_json(ROOT / ('environment_changed_' + str(int(time.time())) + '.json'), environment)
    print('WARNING: package versions differ from restored environment; inspect before mixing results.')
write_json(ROOT / 'environment.json', environment)
(ROOT / 'requirements_snapshot.txt').write_text(
    subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
journal('setup_complete', capa_commit=CAPA_COMMIT)
print(json.dumps(environment, indent=2))

from IPython.display import display, FileLink
def backup():
    # Excludes multi-GB downloads and the pretrained backbone, but includes
    # prepared frames, adaptive checkpoints, results, code, plots and logs.
    archive = shutil.make_archive('/kaggle/working/DA4022_FINAL_BACKUP', 'zip',
                                  root_dir=ROOT.parent, base_dir=ROOT.name)
    print('BACKUP: download this file; /kaggle/working is not durable storage.')
    display(FileLink(archive))
    return archive

print('Setup ready. No model has been loaded yet.')
