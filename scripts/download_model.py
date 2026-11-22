"""Pre-download one Hugging Face model repo into the shared HF cache, so
`serve_vllm.sh` never has to download on first launch. Usage:
    python3 scripts/download_model.py <repo_id>
Run from the vLLM venv (it already depends on huggingface_hub).
"""

import sys

from huggingface_hub import snapshot_download

repo = sys.argv[1]
print("downloading", repo, flush=True)
snapshot_download(repo_id=repo)
print("done", repo, flush=True)
