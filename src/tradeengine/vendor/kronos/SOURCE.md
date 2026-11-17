# Kronos (vendored)

- Upstream: https://github.com/shiyu-coder/Kronos (MIT licence, see LICENSE)
- Commit: 67b630e67f6a18c9e9be918d9b4337c960db1e9a (13 Apr 2026)
- Files: `model/kronos.py`, `model/module.py`. Only change: in `kronos.py` the
  `sys.path.append("../"); from model.module import *` lines became `from .module import *`.
- Weights are NOT stored here: `NeoQuasar/Kronos-small` and `NeoQuasar/Kronos-Tokenizer-base`
  are downloaded from Hugging Face on first use (round 4 Track B, run on Sam's Mac).
- Needs the optional extra: `uv sync --extra kronos` (torch, einops, huggingface_hub, safetensors, tqdm).
  The server never installs it.
